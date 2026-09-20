"""Real Pillow/filesystem tests; only the ComfyUI and Torch boundary is substituted."""

import importlib.util
import math
from pathlib import Path
import random
import sys
import tempfile
import types
import unittest
from unittest.mock import Mock, patch

import numpy as np
from PIL import Image, features


ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("ddht_folder_test", ROOT / "folder_images.py")
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
Node = module.DDHTFolderImage


class Tensor:
    def __init__(self, array):
        self.array = array

    def unsqueeze(self, axis):
        return Tensor(np.expand_dims(self.array, axis))


class Interrupted(BaseException):
    pass


class FolderImageTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.folder = self.root / "图片 with spaces"
        self.folder.mkdir()
        self.temp = self.root / "temp"
        comfy = types.ModuleType("comfy")
        self.management = types.ModuleType("comfy.model_management")
        self.management.throw_exception_if_processing_interrupted = Mock()
        comfy.model_management = self.management
        folders = types.ModuleType("folder_paths")
        folders.get_temp_directory = lambda: str(self.temp)
        torch = types.ModuleType("torch")
        torch.from_numpy = Tensor
        stubs = {"comfy": comfy, "comfy.model_management": self.management,
                 "folder_paths": folders, "torch": torch}
        patcher = patch.dict(sys.modules, stubs)
        patcher.start()
        self.addCleanup(patcher.stop)
        self.node = Node()

    def make_image(self, name="image.png", mode="RGB", color=(30, 60, 90), size=(12, 8), **save_options):
        path = self.folder / name
        path.parent.mkdir(parents=True, exist_ok=True)
        Image.new(mode, size, color).save(path, **save_options)
        return path

    def load(self, **kwargs):
        return self.node.load_image(str(self.folder), **kwargs)

    def preview(self, output):
        descriptor = output["ui"]["images"][0]
        self.assertEqual(descriptor["type"], "temp")
        return self.temp / descriptor["subfolder"] / descriptor["filename"]

    def test_schema_registration_and_always_execute(self):
        self.assertIs(module.NODE_CLASS_MAPPINGS["DDHT_FolderImage"], Node)
        required = Node.INPUT_TYPES()["required"]
        self.assertEqual(required["读取方式"][1]["default"], "随机")
        self.assertTrue(required["seed"][1]["control_after_generate"])
        self.assertFalse(required["填充透明区域"][1]["default"])
        self.assertTrue(Node.OUTPUT_NODE)
        self.assertTrue(math.isnan(Node.IS_CHANGED()))
        self.assertEqual(Node.RETURN_TYPES[:2], ("IMAGE", "MASK"))

    def test_natural_sort_case_insensitive_and_recursive_scan(self):
        for name in ("10.PNG", "2.jpg", "1.png", "child/3.bmp"):
            self.make_image(name)
        (self.folder / "notes.txt").write_text("not an image")
        (self.folder / "fake.png").mkdir()
        self.assertEqual([p.name for p in module.scan_images(self.folder)], ["1.png", "2.jpg", "10.PNG"])
        self.assertEqual(len(module.scan_images(self.folder, True)), 4)
        # Equal natural keys still get a deterministic filename tie-breaker.
        self.make_image("01.png")
        self.assertEqual([p.name for p in module.scan_images(self.folder)][:2], ["01.png", "1.png"])

    def test_quoted_paths_environment_and_empty_folder_errors(self):
        self.assertEqual(module.resolve_folder(f'"{self.folder}"'), self.folder)
        with patch.dict("os.environ", {"DDHT_IMAGE_TEST_DIR": str(self.folder)}):
            self.assertEqual(module.resolve_folder("$DDHT_IMAGE_TEST_DIR"), self.folder)
        for value in ("", " ", None, "\x00", str(self.root / "missing")):
            with self.subTest(value=value), self.assertRaises(ValueError):
                module.resolve_folder(value)
        with self.assertRaisesRegex(ValueError, "没有支持的图片"):
            self.load()

    def test_random_reproducibility_and_global_rng_is_untouched(self):
        for index in range(7):
            self.make_image(f"{index}.png")
        state = random.getstate()
        outputs = [self.load(seed=1234) for _ in range(3)]
        self.assertEqual({out["result"][4] for out in outputs}, {random.Random(1234).randrange(7)})
        self.assertEqual(state, random.getstate())
        self.assertEqual(len({self.load(seed=seed)["result"][4] for seed in range(12)}) > 1, True)
        self.assertNotEqual(self.preview(outputs[0]), self.preview(outputs[1]))

    def test_sequence_wraps_and_seed_does_not_reset_it(self):
        for name in ("1.png", "10.png", "2.png"):
            self.make_image(name)
        outputs = [self.load(读取方式="顺序", seed=seed)["result"] for seed in range(5)]
        self.assertEqual([out[2] for out in outputs], ["1.png", "2.png", "10.png", "1.png", "2.png"])
        self.assertEqual([out[4] for out in outputs], [0, 1, 2, 0, 1])
        self.assertTrue(all(out[5] == 3 for out in outputs))

    def test_sequence_start_reset_file_list_and_mode_changes(self):
        for name in ("1.png", "2.png", "3.png"):
            self.make_image(name)
        def selected(**options):
            return self.load(读取方式="顺序", 顺序起始索引=4, **options)["result"][4]
        self.assertEqual([selected(), selected()], [1, 2])
        self.assertEqual(selected(顺序重置标记=1), 1)
        self.assertEqual(selected(顺序重置标记=1), 2)
        self.make_image("4.png")
        self.assertEqual(selected(顺序重置标记=1), 0)  # New list; start 4 modulo 4.
        self.load(seed=0)
        self.assertEqual(selected(顺序重置标记=1), 0)

    def test_separate_nodes_and_folder_changes_have_independent_positions(self):
        self.make_image("1.png")
        self.make_image("2.png")
        self.assertEqual(self.load(读取方式="顺序")["result"][4], 0)
        other = Node()
        self.assertEqual(other.load_image(str(self.folder), 读取方式="顺序")["result"][4], 0)
        self.assertEqual(self.load(读取方式="顺序")["result"][4], 1)
        nested = self.make_image("child/a.png").parent
        self.node.load_image(str(nested), 读取方式="顺序")
        self.assertEqual(self.load(读取方式="顺序")["result"][4], 0)

    def test_failed_read_or_preview_does_not_advance_sequence(self):
        self.make_image("1.png")
        second = self.folder / "2.png"
        second.write_bytes(b"broken PNG")
        self.load(读取方式="顺序")
        with self.assertRaisesRegex(RuntimeError, "无法解码"):
            self.load(读取方式="顺序")
        self.make_image("2.png")
        with patch.object(module, "save_preview", side_effect=OSError("disk full")), self.assertRaises(OSError):
            self.load(读取方式="顺序")
        self.assertEqual(self.load(读取方式="顺序")["result"][4], 1)

    def test_cancelled_read_does_not_advance(self):
        self.make_image("1.png")
        self.make_image("2.png")
        with patch.object(module, "read_image", side_effect=Interrupted), self.assertRaises(Interrupted):
            self.load(读取方式="顺序")
        self.assertEqual(self.load(读取方式="顺序")["result"][4], 0)

    def test_rgba_mask_preview_and_fill_use_same_pixels(self):
        path = self.folder / "alpha.png"
        pixels = np.array([[[200, 20, 40, 0], [200, 20, 40, 128], [200, 20, 40, 255]]], dtype=np.uint8)
        Image.fromarray(pixels).save(path)
        original = path.read_bytes()
        output = self.load()
        rgb, mask = (out.array for out in output["result"][:2])
        self.assertEqual(rgb.shape, (1, 1, 3, 3))
        self.assertEqual(mask.shape, (1, 1, 3))
        self.assertEqual(rgb.dtype, np.float32)
        np.testing.assert_allclose(mask[0, 0], [1, 127 / 255, 0], atol=1e-7)
        with Image.open(self.preview(output)) as preview:
            np.testing.assert_array_equal(np.asarray(preview), pixels)
        filled = self.load(填充透明区域=True, 填充颜色="#003366")
        np.testing.assert_array_equal(filled["result"][1].array, np.zeros((1, 1, 3)))
        rgb = filled["result"][0].array[0]
        np.testing.assert_allclose(rgb[0, 0] * 255, [0, 51, 102], atol=1e-5)
        np.testing.assert_allclose(rgb[0, 1] * 255, [100, 35, 71], atol=1e-5)
        with Image.open(self.preview(filled)) as preview:
            self.assertEqual(preview.mode, "RGB")
            np.testing.assert_array_equal(np.asarray(preview), np.rint(rgb * 255).astype(np.uint8))
        self.assertEqual(path.read_bytes(), original)

    def test_palette_and_grayscale_alpha_transparency(self):
        image = Image.new("P", (2, 1))
        image.putpalette([255, 0, 0, 0, 255, 0] + [0] * 762)
        image.putdata([0, 1])
        image.save(self.folder / "palette.png", transparency=0)
        _, mask, _ = module.read_image(self.folder / "palette.png")
        np.testing.assert_array_equal(mask, [[1, 0]])
        path = self.make_image("gray.png", mode="LA", color=(127, 64))
        rgb, mask, _ = module.read_image(path)
        np.testing.assert_allclose(rgb, 127 / 255)
        np.testing.assert_allclose(mask, 191 / 255)

    def test_common_formats_and_opaque_masks(self):
        names = ["jpg", "jpeg", "jfif", "png", "bmp", "gif", "tiff", "tif", "ico", "tga", "ppm", "pgm", "pbm", "pcx"]
        if features.check("webp"):
            names.append("webp")
        if features.check("avif"):
            names.append("avif")
        if features.check("jpg_2000"):
            names.append("jp2")
        for extension in names:
            with self.subTest(extension=extension):
                mode, color = ("L", 100) if extension in {"pgm", "pbm"} else ("RGB", (30, 60, 90))
                path = self.make_image(f"format.{extension}", mode=mode, color=color, size=(32, 24))
                rgb, mask, _ = module.read_image(path)
                self.assertEqual(rgb.shape[-1], 3)
                self.assertEqual(rgb.dtype, np.float32)
                self.assertEqual(mask.dtype, np.float32)
                np.testing.assert_array_equal(mask, np.zeros(mask.shape))

    def test_multiframe_files_return_first_frame_only(self):
        first, second = Image.new("RGB", (8, 6), "red"), Image.new("RGB", (8, 6), "blue")
        for extension in ("gif", "png", "tiff", "webp"):
            with self.subTest(extension=extension):
                path = self.folder / f"animated.{extension}"
                first.save(path, save_all=True, append_images=[second], duration=100, lossless=True)
                rgb, _, _ = module.read_image(path)
                self.assertEqual(rgb.shape, (6, 8, 3))
                np.testing.assert_array_equal(rgb[0, 0], [1, 0, 0])

    def test_exif_orientation_is_applied_to_image_and_mask(self):
        image = Image.new("RGBA", (4, 2), (30, 60, 90, 0))
        image.putpixel((0, 0), (255, 0, 0, 255))
        exif = Image.Exif()
        exif[274] = 6
        path = self.folder / "rotated.png"
        image.save(path, exif=exif)
        rgb, mask, preview = module.read_image(path)
        self.assertEqual(rgb.shape, (4, 2, 3))
        np.testing.assert_array_equal(rgb[0, 1], [1, 0, 0])
        self.assertEqual(mask[0, 1], 0)
        self.assertEqual(preview.info, {})

    def test_sixteen_bit_png_keeps_middle_gray(self):
        path = self.folder / "gray16.png"
        Image.fromarray(np.array([[0, 32768, 65535]], dtype=np.uint16)).save(path)
        rgb, mask, _ = module.read_image(path)
        np.testing.assert_allclose(rgb[0, :, 0], [0, 32768 / 65535, 1])
        np.testing.assert_array_equal(mask, [[0, 0, 0]])

    def test_sixteen_bit_png_transparent_sample(self):
        path = self.folder / "transparent16.png"
        Image.fromarray(np.array([[0, 32768, 65535]], dtype=np.uint16)).save(path, transparency=32768)
        _, mask, preview = module.read_image(path)
        np.testing.assert_array_equal(mask, [[0, 1, 0]])
        self.assertEqual(preview.getpixel((1, 0))[3], 0)
        rgb, mask, _ = module.read_image(path, True, (0, 255, 0))
        np.testing.assert_array_equal(rgb[0, 1], [0, 1, 0])
        np.testing.assert_array_equal(mask, [[0, 0, 0]])

    def test_cmyk_and_jpeg_exif_orientation(self):
        exif = Image.Exif()
        exif[274] = 6
        path = self.make_image("cmyk.jpg", mode="CMYK", color=(0, 255, 255, 0), exif=exif)
        rgb, _, _ = module.read_image(path)
        self.assertEqual(rgb.shape, (12, 8, 3))
        np.testing.assert_allclose(rgb[0, 0], [1, 0, 0], atol=0.02)

    def test_thumbnail_does_not_resize_output_or_copy_metadata(self):
        self.make_image(size=(1800, 1200))
        output = self.load()
        self.assertEqual(output["result"][0].array.shape, (1, 1200, 1800, 3))
        with Image.open(self.preview(output)) as preview:
            self.assertEqual(preview.size, (1024, 683))
            self.assertNotIn("prompt", preview.info)
        self.assertEqual(len(list(self.folder.iterdir())), 1)

    def test_bad_parameters_fail_before_preview(self):
        self.make_image()
        self.assertEqual(module.parse_color(" #0f8 "), (0, 255, 136))
        for value in ("red", "#12", "#GGGGGG", "#ffffff00", None):
            with self.subTest(color=value), self.assertRaises(ValueError):
                self.load(填充透明区域=True, 填充颜色=value)
        for options in ({"seed": -1}, {"seed": True}, {"seed": 2**64}, {"seed": 0.5},
                        {"读取方式": "other"}, {"包含子文件夹": "false"}, {"填充透明区域": "false"},
                        {"顺序起始索引": -1}, {"顺序重置标记": -1}):
            with self.subTest(options=options), self.assertRaises(ValueError):
                self.load(**options)
        self.assertFalse(self.temp.exists())
        self.load(填充透明区域=False, 填充颜色="ignored")

    def test_missing_optional_decoder_has_actionable_error(self):
        path = self.folder / "photo.heic"
        with patch.dict(Image.OPEN, {}, clear=True), patch.object(Image, "init"), \
                patch.object(module.importlib, "import_module", side_effect=ImportError), \
                self.assertRaisesRegex(RuntimeError, "pillow-heif"):
            module.read_image(path)

    def test_real_package_entry_registers_folder_node(self):
        name = "ddht_folder_package_test"
        stubs = {f"{name}.folder_images": module}
        for child in ("clip_management", "deepseek_api", "local_llm", "nodes", "sla_attention", "text_validation"):
            stub = types.ModuleType(f"{name}.{child}")
            stub.NODE_CLASS_MAPPINGS = {child: object()}
            stub.NODE_DISPLAY_NAME_MAPPINGS = {child: child}
            stubs[stub.__name__] = stub
        package_spec = importlib.util.spec_from_file_location(name, ROOT / "__init__.py", submodule_search_locations=[str(ROOT)])
        package = importlib.util.module_from_spec(package_spec)
        stubs[name] = package
        with patch.dict(sys.modules, stubs):
            package_spec.loader.exec_module(package)
        self.assertIs(package.NODE_CLASS_MAPPINGS["DDHT_FolderImage"], Node)
        self.assertEqual(len(package.NODE_CLASS_MAPPINGS), 7)


if __name__ == "__main__":
    unittest.main()
