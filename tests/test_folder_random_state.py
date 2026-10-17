"""Real JSON, hashing, locks and subprocess persistence; no GPU required."""

import importlib
import json
import os
from pathlib import Path
import random
import subprocess
import sys
import tempfile
import types
import unittest
from unittest.mock import Mock, patch

import numpy as np
from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
PACKAGE = "ddht_weighted_test"
package = types.ModuleType(PACKAGE)
package.__path__ = [str(ROOT)]
sys.modules[PACKAGE] = package
state_module = importlib.import_module(f"{PACKAGE}.folder_random_state")
nodes = importlib.import_module(f"{PACKAGE}.folder_images")


class Interrupted(BaseException):
    pass


class Tensor:
    def __init__(self, array):
        self.array = array

    def unsqueeze(self, axis):
        return Tensor(np.expand_dims(self.array, axis))


class WeightedFolderTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.folder = self.root / "图片"
        self.folder.mkdir()
        self.user = self.root / "user"
        self.documents = self.user / "ddht" / "folder_image_weights"
        self.preview_dir = self.root / "preview"
        for index in range(3):
            self.image(f"{index}.png", (index * 50, 80, 120))

    def image(self, name, color=(10, 20, 30)):
        path = self.folder / name
        path.parent.mkdir(parents=True, exist_ok=True)
        Image.new("RGB", (8, 6), color).save(path)
        return path

    def run_draw(self, seed=0, reset=0, recursive=False, consume=None, folder=None, documents=None, check=lambda: None):
        folder = folder or self.folder
        files = nodes.scan_images(folder, recursive, allow_empty=True)
        return state_module.run_weighted(folder, files, recursive, seed, reset, documents or self.documents,
                                         consume or (lambda index: index), check)

    def document_path(self):
        return state_module.state_path_for(self.folder, self.documents)

    def document(self):
        return json.loads(self.document_path().read_text(encoding="utf-8"))

    def assert_document_consistent(self, state):
        total = state["total_draws"]
        self.assertEqual(sum(item["use_count"] for item in state["images"]), total)
        self.assertEqual([item["rank"] for item in state["images"]], list(range(1, len(state["images"]) + 1)))
        weights = [item["weight"] for item in state["images"]]
        self.assertEqual(weights, sorted(weights, reverse=True))
        for item in state["images"]:
            self.assertEqual(item["weight"], total - item["last_used_draw"] + 1)
            self.assertEqual(len(item["sha256"]), 64)

    def test_document_ranks_every_image_and_updates_only_selected_history(self):
        index, path = self.run_draw()
        state = self.document()
        self.assertEqual(path, str(self.document_path()))
        self.assertEqual(state["total_draws"], 1)
        self.assertEqual(state["image_count"], 3)
        selected = next(item for item in state["images"] if item["relative_path"] == f"{index}.png")
        self.assertEqual((selected["weight"], selected["use_count"], selected["last_used_draw"]), (1, 1, 1))
        self.assertIsNotNone(selected["last_used_at"])
        self.assertEqual([item["weight"] for item in state["images"]], [2, 2, 1])
        self.assert_document_consistent(state)
        self.run_draw(seed=123)
        self.assertEqual(self.document()["generation_id"], state["generation_id"])
        self.assert_document_consistent(self.document())

    def test_fixed_seed_reproduces_history_and_does_not_touch_global_rng(self):
        state = random.getstate()
        first = [self.run_draw(seed=123)[0] for _ in range(30)]
        second = [self.run_draw(seed=123, documents=self.root / "other-state")[0] for _ in range(30)]
        self.assertEqual(first, second)
        self.assertGreater(len(set(first)), 1)
        self.assertEqual(state, random.getstate())
        self.assertEqual(self.document()["total_draws"], 30)

    def test_weighted_ticket_intervals_have_exact_requested_proportions(self):
        manifest = [{"relative_path": name} for name in ("a", "b", "c")]
        state = {"total_draws": 10, "images": [{"relative_path": name, "weight": weight}
                                                for name, weight in zip(("a", "b", "c"), (1, 3, 6))]}
        picked = []
        for ticket in range(10):
            rng = Mock()
            rng.randrange.return_value = ticket
            with patch.object(state_module.random, "Random", return_value=rng):
                picked.append(state_module._weighted_index(state, manifest, 0))
            rng.randrange.assert_called_once_with(10)
        self.assertEqual([picked.count(index) for index in range(3)], [1, 3, 6])

    def test_file_add_remove_rename_and_scope_changes_rebuild_same_document(self):
        self.run_draw()
        previous = self.document()["generation_id"]
        changes = [lambda: self.image("new.png"),
                   lambda: (self.folder / "new.png").unlink(),
                   lambda: (self.folder / "0.png").rename(self.folder / "renamed.png")]
        for change in changes:
            change()
            self.run_draw()
            state = self.document()
            self.assertNotEqual(state["generation_id"], previous)
            self.assertEqual(state["total_draws"], 1)
            self.assertEqual(state["rebuild_reason"], "image_list_or_content_changed")
            previous = state["generation_id"]
        self.image("child/child.png")
        self.run_draw(recursive=True)
        self.assertNotEqual(self.document()["generation_id"], previous)
        self.assertEqual(self.document()["image_count"], 4)
        self.assertEqual(len(list(self.documents.glob("*.json"))), 1)

    def test_same_name_size_and_mtime_content_change_rebuilds(self):
        path = self.folder / "0.png"
        path.write_bytes(b"AAAA")
        self.run_draw()
        previous = self.document()["generation_id"]
        before = path.stat()
        path.write_bytes(b"BBBB")
        os.utime(path, ns=(before.st_atime_ns, before.st_mtime_ns))
        self.assertEqual(path.stat().st_size, before.st_size)
        self.assertEqual(path.stat().st_mtime_ns, before.st_mtime_ns)
        self.run_draw()
        self.assertNotEqual(self.document()["generation_id"], previous)
        self.assertEqual(self.document()["total_draws"], 1)

    def test_non_image_file_and_same_bytes_rewrite_do_not_reset(self):
        self.run_draw()
        previous = self.document()["generation_id"]
        (self.folder / "notes.txt").write_text("new note")
        path = self.folder / "0.png"
        path.write_bytes(path.read_bytes())
        self.run_draw()
        self.assertEqual(self.document()["generation_id"], previous)
        self.assertEqual(self.document()["total_draws"], 2)

    def test_reset_marker_creates_new_generation_but_seed_change_does_not(self):
        self.run_draw()
        previous = self.document()["generation_id"]
        self.run_draw(seed=8)
        self.assertEqual(self.document()["generation_id"], previous)
        self.run_draw(seed=8, reset=1)
        self.assertNotEqual(self.document()["generation_id"], previous)
        self.assertEqual(self.document()["total_draws"], 1)
        self.assertEqual(self.document()["rebuild_reason"], "manual_reset")

    def test_each_folder_gets_its_own_document(self):
        self.run_draw()
        other = self.root / "other"
        other.mkdir()
        Image.new("RGB", (8, 6), "red").save(other / "image.png")
        _, other_path = self.run_draw(folder=other)
        self.assertNotEqual(str(self.document_path()), other_path)
        self.assertEqual(len(list(self.documents.glob("*.json"))), 2)
        self.assertEqual(self.document()["total_draws"], 1)
        self.assertEqual(state_module.state_path_for(self.folder / ".", self.documents), self.document_path())

    def test_first_failed_use_creates_document_without_consuming_image(self):
        with self.assertRaisesRegex(RuntimeError, "decode failed"):
            self.run_draw(consume=Mock(side_effect=RuntimeError("decode failed")))
        self.assertEqual(self.document()["total_draws"], 0)
        self.assertEqual([item["weight"] for item in self.document()["images"]], [1, 1, 1])

    def test_failed_or_cancelled_use_leaves_existing_document_unchanged(self):
        self.run_draw()
        before = self.document_path().read_bytes()
        for error in (RuntimeError("preview failed"), Interrupted()):
            with self.subTest(error=type(error).__name__), self.assertRaises(type(error)):
                self.run_draw(consume=Mock(side_effect=error))
            self.assertEqual(self.document_path().read_bytes(), before)

    def test_write_failure_does_not_return_success_or_corrupt_document(self):
        self.run_draw()
        before = self.document_path().read_bytes()
        with patch.object(state_module.os, "replace", side_effect=OSError("disk full")), self.assertRaisesRegex(RuntimeError, "无法保存"):
            self.run_draw()
        self.assertEqual(self.document_path().read_bytes(), before)
        self.assertFalse(list(self.documents.glob("*.tmp")))

    def test_corrupt_document_is_rebuilt(self):
        self.run_draw()
        self.document_path().write_text("{broken", encoding="utf-8")
        self.run_draw()
        self.assertEqual(self.document()["total_draws"], 1)
        state = self.document()
        state["images"][0]["weight"] = 999
        previous = state["generation_id"]
        self.document_path().write_text(json.dumps(state), encoding="utf-8")
        self.run_draw()
        self.assertNotEqual(self.document()["generation_id"], previous)
        self.assert_document_consistent(self.document())

    def test_image_modified_during_consume_is_not_committed(self):
        self.run_draw()
        before = self.document_path().read_bytes()
        def consume(index):
            files = nodes.scan_images(self.folder)
            files[index].write_bytes(b"changed during decode")
        with self.assertRaisesRegex(RuntimeError, "读取期间被修改"):
            self.run_draw(consume=consume)
        self.assertEqual(self.document_path().read_bytes(), before)
        self.run_draw()
        self.assertEqual(self.document()["total_draws"], 1)

    def test_empty_folder_rebuilds_empty_document_and_single_file_works(self):
        self.run_draw()
        for path in self.folder.glob("*.png"):
            path.unlink()
        with self.assertRaisesRegex(ValueError, "为空列表"):
            self.run_draw()
        self.assertEqual(self.document()["images"], [])
        self.assertEqual(self.document()["total_draws"], 0)
        self.image("only.png")
        for _ in range(4):
            self.assertEqual(self.run_draw()[0], 0)
        self.assertEqual(self.document()["images"][0]["weight"], 1)
        self.assertEqual(self.document()["total_draws"], 4)

    def test_progress_survives_processes_and_concurrent_updates(self):
        self.run_draw()
        script = '''
import importlib.util, sys
from pathlib import Path
spec = importlib.util.spec_from_file_location("weighted", sys.argv[1])
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)
folder, documents = Path(sys.argv[2]), Path(sys.argv[3])
files = sorted(folder.glob("*.png"))
for _ in range(8):
    m.run_weighted(folder, files, False, 0, 0, documents, lambda index: index)
'''
        processes = [subprocess.Popen([sys.executable, "-c", script, str(ROOT / "folder_random_state.py"),
                                       str(self.folder), str(self.documents)], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
                     for _ in range(3)]
        try:
            for process in processes:
                stdout, stderr = process.communicate(timeout=20)
                self.assertEqual(process.returncode, 0, stderr.decode())
        finally:
            for process in processes:
                if process.poll() is None:
                    process.kill()
                    process.wait()
        self.assertEqual(self.document()["total_draws"], 25)
        self.assert_document_consistent(self.document())

    def test_node_outputs_document_path_and_recovers_after_instance_recreation(self):
        comfy = types.ModuleType("comfy")
        management = types.ModuleType("comfy.model_management")
        management.throw_exception_if_processing_interrupted = lambda: None
        comfy.model_management = management
        folders = types.ModuleType("folder_paths")
        folders.get_temp_directory = lambda: str(self.preview_dir)
        folders.get_user_directory = lambda: str(self.user)
        torch = types.ModuleType("torch")
        torch.from_numpy = Tensor
        with patch.dict(sys.modules, {"comfy": comfy, "comfy.model_management": management,
                                      "folder_paths": folders, "torch": torch}):
            for _ in range(3):
                output = nodes.DDHTFolderImage().load_image(str(self.folder), 读取方式="时间间隔加权随机")
                self.assertEqual(output["result"][6], str(self.document_path()))
                self.assertEqual(output["result"][0].array.shape, (1, 6, 8, 3))
                self.assertEqual(output["ui"]["images"][0]["type"], "temp")
            before = self.document_path().read_bytes()
            with patch.object(nodes, "save_preview", side_effect=RuntimeError("preview failed")), self.assertRaises(RuntimeError):
                nodes.DDHTFolderImage().load_image(str(self.folder), 读取方式="时间间隔加权随机")
            self.assertEqual(self.document_path().read_bytes(), before)
        self.assertEqual(self.document()["total_draws"], 3)


if __name__ == "__main__":
    unittest.main()
