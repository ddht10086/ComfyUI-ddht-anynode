"""Native loader delegation and real NumPy/Pillow alpha/preview verification."""

import importlib
import importlib.util
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest.mock import Mock, patch

import numpy as np
from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
PACKAGE = "ddht_load_image_test"
package = types.ModuleType(PACKAGE)
package.__path__ = [str(ROOT)]
sys.modules[PACKAGE] = package
module = importlib.import_module(f"{PACKAGE}.load_image")
Node = module.DDHTLoadImage


class Tensor(np.lib.mixins.NDArrayOperatorsMixin):
    """Small numerical tensor boundary; pixel arithmetic is real NumPy."""
    def __init__(self, array):
        self.array = np.asarray(array)

    @property
    def shape(self):
        return self.array.shape

    @property
    def dtype(self):
        return self.array.dtype

    @property
    def device(self):
        return "cpu"

    def __array_ufunc__(self, ufunc, method, *inputs, **kwargs):
        values = [value.array if isinstance(value, Tensor) else value for value in inputs]
        return Tensor(getattr(ufunc, method)(*values, **kwargs))

    def __getitem__(self, key):
        return Tensor(self.array[key])

    def to(self, device=None, dtype=None):
        return Tensor(self.array.astype(dtype or self.dtype))

    def new_tensor(self, values):
        return Tensor(np.array(values, dtype=self.dtype))

    def new_zeros(self, shape):
        return Tensor(np.zeros(shape, dtype=self.dtype))

    def unsqueeze(self, axis):
        return Tensor(np.expand_dims(self.array, axis))

    def any(self):
        return self.array.any()

    def detach(self):
        return self

    def float(self):
        return self.to(dtype=np.float32)

    def cpu(self):
        return self

    def numpy(self):
        return self.array


class LoadImageTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.native = Mock()
        self.native.INPUT_TYPES.return_value = {"required": {
            "image": (["alpha.png", "photo.jpg"], {"image_upload": True})}}
        self.native.VALIDATE_INPUTS.return_value = True
        self.native.IS_CHANGED.return_value = "native-file-digest"
        core = types.ModuleType("nodes")
        core.LoadImage = self.native
        folders = types.ModuleType("folder_paths")
        folders.get_temp_directory = lambda: str(self.root)
        comfy = types.ModuleType("comfy")
        comfy.model_management = types.SimpleNamespace(throw_exception_if_processing_interrupted=Mock())
        self.interrupt = comfy.model_management.throw_exception_if_processing_interrupted
        patcher = patch.dict(sys.modules, {"nodes": core, "folder_paths": folders, "comfy": comfy})
        patcher.start()
        self.addCleanup(patcher.stop)
        self.images = Tensor(np.array([[[[1, 0, 0], [1, 0, 0], [1, 0, 0]]]], dtype=np.float32))
        self.masks = Tensor(np.array([[[1, 0.5, 0]]], dtype=np.float32))
        self.native.return_value.load_image.return_value = (self.images, self.masks)

    def load(self, **options):
        return Node().load_image("alpha.png", **options)

    def preview(self, output):
        entry = output["ui"]["images"][0]
        self.assertEqual(entry["type"], "temp")
        return Image.open(self.root / entry["subfolder"] / entry["filename"])

    def test_upload_schema_is_preserved_without_mutating_native_schema(self):
        schema = Node.INPUT_TYPES()
        self.assertTrue(schema["required"]["image"][1]["image_upload"])
        self.assertEqual(schema["required"]["image"][0], ["alpha.png", "photo.jpg"])
        self.assertFalse(schema["required"]["填充透明区域"][1]["default"])
        self.assertEqual(schema["required"]["填充颜色"][1]["default"], "#FFFFFF")
        self.assertEqual(list(self.native.INPUT_TYPES.return_value["required"]), ["image"])

    def test_disabled_returns_native_tensors_unchanged_and_alpha_preview(self):
        output = self.load(填充颜色="ignored while disabled")
        self.native.return_value.load_image.assert_called_once_with("alpha.png")
        self.assertIs(output["result"][0], self.images)
        self.assertIs(output["result"][1], self.masks)
        with self.preview(output) as image:
            self.assertEqual(image.mode, "RGBA")
            self.assertEqual([image.getpixel((x, 0))[3] for x in range(3)], [0, 128, 255])

    def test_white_fill_blends_partial_alpha_and_clears_mask(self):
        before_images, before_masks = self.images.array.copy(), self.masks.array.copy()
        output = self.load(填充透明区域=True)
        images, masks = output["result"]
        np.testing.assert_allclose(images.array[0, 0], [[1, 1, 1], [1, 0.5, 0.5], [1, 0, 0]])
        np.testing.assert_array_equal(masks.array, 0)
        np.testing.assert_array_equal(self.images.array, before_images)
        np.testing.assert_array_equal(self.masks.array, before_masks)
        with self.preview(output) as image:
            self.assertEqual(image.mode, "RGB")
            self.assertEqual(image.getpixel((1, 0)), (255, 128, 128))

    def test_custom_colors_and_short_hex(self):
        for color, expected in (("#000000", [0, 0, 0]), (" #0f8 ", [0, 1, 136 / 255])):
            with self.subTest(color=color):
                images, masks = self.load(填充透明区域=True, 填充颜色=color)["result"]
                np.testing.assert_allclose(images.array[0, 0, 0], expected)
                np.testing.assert_array_equal(masks.array, 0)

    def test_multiframe_batch_and_dtype_are_preserved(self):
        self.images = Tensor(np.array([[[[1, 0, 0]]], [[[0, 0, 1]]]], dtype=np.float16))
        self.masks = Tensor(np.array([[[0.25]], [[0.75]]], dtype=np.float16))
        self.native.return_value.load_image.return_value = (self.images, self.masks)
        output = self.load(填充透明区域=True, 填充颜色="#0F0")
        images, masks = output["result"]
        self.assertEqual(images.shape, (2, 1, 1, 3))
        self.assertEqual(images.dtype, np.float16)
        self.assertEqual(masks.dtype, np.float16)
        np.testing.assert_array_equal(images.array[:, 0, 0], [[0.75, 0.25, 0], [0, 0.75, 0.25]])
        self.assertEqual(len(output["ui"]["images"]), 1)

    def test_native_opaque_64_pixel_zero_mask_is_not_broadcast(self):
        self.images = Tensor(np.full((1, 10, 20, 3), 0.3, dtype=np.float32))
        self.masks = Tensor(np.zeros((1, 64, 64), dtype=np.float32))
        self.native.return_value.load_image.return_value = (self.images, self.masks)
        images, masks = self.load(填充透明区域=True)["result"]
        self.assertIs(images, self.images)
        self.assertIs(masks, self.masks)

    def test_nonzero_mismatched_mask_is_reported(self):
        self.native.return_value.load_image.return_value = (self.images, Tensor(np.ones((1, 64, 64))))
        with self.assertRaisesRegex(ValueError, "尺寸不一致"):
            self.load(填充透明区域=True)

    def test_validation_delegates_and_rejects_invalid_options_before_reading(self):
        self.assertTrue(Node.VALIDATE_INPUTS("alpha.png"))
        self.native.VALIDATE_INPUTS.assert_called_once_with("alpha.png")
        self.native.VALIDATE_INPUTS.return_value = "Invalid image file: missing.png"
        self.assertEqual(Node.VALIDATE_INPUTS("missing.png"), "Invalid image file: missing.png")
        for options in ({"填充透明区域": "false"}, {"填充透明区域": True, "填充颜色": "red"},
                        {"填充透明区域": True, "填充颜色": "#12345678"}):
            with self.subTest(options=options):
                self.assertIsInstance(Node.VALIDATE_INPUTS("alpha.png", **options), str)
                with self.assertRaises(ValueError):
                    self.load(**options)
        self.native.return_value.load_image.assert_not_called()

    def test_file_change_detection_uses_native_loader(self):
        self.assertEqual(Node.IS_CHANGED("alpha.png", True, "#000"), "native-file-digest")
        self.native.IS_CHANGED.return_value = "new-file-digest"
        self.assertEqual(Node.IS_CHANGED("alpha.png"), "new-file-digest")
        self.native.IS_CHANGED.assert_called_with("alpha.png")

    def test_native_read_errors_and_cancellation_propagate(self):
        self.native.return_value.load_image.side_effect = RuntimeError("native decode failed")
        with self.assertRaisesRegex(RuntimeError, "native decode failed"):
            self.load()
        self.native.return_value.load_image.reset_mock()
        self.interrupt.side_effect = RuntimeError("cancelled")
        with self.assertRaisesRegex(RuntimeError, "cancelled"):
            self.load()
        self.native.return_value.load_image.assert_not_called()

    def test_preview_is_thumbnail_without_resizing_outputs(self):
        self.images = Tensor(np.full((1, 1200, 1800, 3), 0.5, dtype=np.float32))
        self.masks = Tensor(np.zeros((1, 64, 64), dtype=np.float32))
        self.native.return_value.load_image.return_value = (self.images, self.masks)
        output = self.load()
        self.assertEqual(output["result"][0].shape, (1, 1200, 1800, 3))
        with self.preview(output) as image:
            self.assertEqual(image.size, (1024, 683))

    def test_package_entry_registers_new_node_and_preserves_other_nodes(self):
        name = "ddht_load_image_package_test"
        stubs = {f"{name}.load_image": module}
        others = ("clip_management", "deepseek_api", "folder_images", "local_llm", "nodes", "sla_attention", "text_validation")
        for child in others:
            stub = types.ModuleType(f"{name}.{child}")
            stub.NODE_CLASS_MAPPINGS = {child: object()}
            stub.NODE_DISPLAY_NAME_MAPPINGS = {child: child}
            stubs[stub.__name__] = stub
        spec = importlib.util.spec_from_file_location(name, ROOT / "__init__.py", submodule_search_locations=[str(ROOT)])
        package = importlib.util.module_from_spec(spec)
        stubs[name] = package
        with patch.dict(sys.modules, stubs):
            spec.loader.exec_module(package)
        self.assertIs(package.NODE_CLASS_MAPPINGS["DDHT_LoadImage"], Node)
        self.assertTrue(set(others).issubset(package.NODE_CLASS_MAPPINGS))
        self.assertEqual(package.NODE_DISPLAY_NAME_MAPPINGS["DDHT_LoadImage"], "读取图片（透明填充）- DDHT")
        self.assertEqual(Node.RETURN_TYPES, ("IMAGE", "MASK"))


if __name__ == "__main__":
    unittest.main()
