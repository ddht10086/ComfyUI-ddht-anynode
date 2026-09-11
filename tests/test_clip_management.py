"""Dependency-light tests for the DDHT CLIP unload node."""

from __future__ import annotations

import importlib.util
from pathlib import Path
import sys
import types
import unittest
import uuid


_ROOT = Path(__file__).resolve().parents[1]


class FakeDevice:
    def __init__(self, device_type):
        self.type = device_type

    def __str__(self):
        return self.type


class FakePatcher:
    def __init__(self, loaded=512 * 1024 * 1024, family=None, device="cpu"):
        self._loaded = loaded
        self.clone_base_uuid = family or uuid.uuid4()
        self.offload_device = FakeDevice(device)
        self.model = object()
        self.patch_target = None
        self.partial_target = None
        self.detached = False

    def loaded_size(self):
        return self._loaded

    def model_patches_to(self, device):
        self.patch_target = device

    def partially_unload(self, device, _amount):
        self.partial_target = device
        freed = self._loaded
        self._loaded = 0
        return freed

    def detach(self):
        self.detached = True
        self._loaded = 0


class FakeCLIP:
    def __init__(self, patcher):
        self.patcher = patcher


def _load_node_module():
    comfy = types.ModuleType("comfy")
    comfy.__path__ = []
    management = types.ModuleType("comfy.model_management")
    management._loaded = []
    management.empty_calls = 0
    management.full_calls = []
    management.loaded_models = lambda: list(management._loaded)

    def empty_cache(force=False):
        management.empty_calls += 1
        management.last_force = force

    def full_unload(patcher, **kwargs):
        management.full_calls.append((patcher, kwargs))
        for candidate in management._loaded:
            if candidate.clone_base_uuid == patcher.clone_base_uuid:
                candidate._loaded = 0
        management._loaded = [
            candidate
            for candidate in management._loaded
            if candidate.clone_base_uuid != patcher.clone_base_uuid
        ]

    management.soft_empty_cache = empty_cache
    management.unload_model_and_clones = full_unload
    comfy.model_management = management
    sys.modules["comfy"] = comfy
    sys.modules["comfy.model_management"] = management

    spec = importlib.util.spec_from_file_location(
        "ddht_clip_management_test", _ROOT / "clip_management.py"
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module, management


clip_module, fake_management = _load_node_module()


class CLIPManagementTests(unittest.TestCase):
    def setUp(self):
        fake_management._loaded = []
        fake_management.empty_calls = 0
        fake_management.full_calls = []

    def test_registration_and_ordering_schema(self):
        node_class = clip_module.NODE_CLASS_MAPPINGS["DDHT_UnloadCLIP"]
        schema = node_class.INPUT_TYPES()
        self.assertEqual(node_class.RETURN_TYPES[:2], ("CONDITIONING", "CONDITIONING"))
        self.assertIn("clip", schema["required"])
        self.assertIn("conditioning_1", schema["required"])
        self.assertIn("conditioning_2", schema["optional"])
        self.assertNotEqual(node_class.IS_CHANGED(), node_class.IS_CHANGED())

    def test_cpu_mode_offloads_clip_and_preserves_conditioning_identity(self):
        patcher = FakePatcher()
        fake_management._loaded = [patcher]
        clip = FakeCLIP(patcher)
        positive = object()
        negative = object()

        result = clip_module.DDHTUnloadCLIP().unload_clip(
            clip, positive, clip_module.MODE_CPU, negative
        )

        self.assertIs(result[0], positive)
        self.assertIs(result[1], negative)
        self.assertEqual(patcher._loaded, 0)
        self.assertEqual(patcher.partial_target.type, "cpu")
        self.assertEqual(fake_management.empty_calls, 1)
        self.assertIn("512.0 MB", result[2])

    def test_cpu_mode_deduplicates_clones_sharing_one_model(self):
        family = uuid.uuid4()
        first = FakePatcher(family=family)
        second = FakePatcher(family=family)
        second.model = first.model
        fake_management._loaded = [first, second]

        clip_module.DDHTUnloadCLIP().unload_clip(
            FakeCLIP(first), object(), clip_module.MODE_CPU
        )

        self.assertEqual(first._loaded, 0)
        self.assertEqual(second._loaded, 512 * 1024 * 1024)

    def test_full_mode_uses_official_all_device_unload(self):
        patcher = FakePatcher()
        fake_management._loaded = [patcher]

        result = clip_module.DDHTUnloadCLIP().unload_clip(
            FakeCLIP(patcher), object(), clip_module.MODE_FULL
        )

        self.assertEqual(len(fake_management.full_calls), 1)
        self.assertTrue(fake_management.full_calls[0][1]["all_devices"])
        self.assertEqual(fake_management.empty_calls, 1)
        self.assertIn("512.0 MB", result[2])

    def test_cpu_mode_rejects_gpu_only_offload_device(self):
        patcher = FakePatcher(device="cuda")
        fake_management._loaded = [patcher]

        with self.assertRaisesRegex(RuntimeError, "--gpu-only"):
            clip_module.DDHTUnloadCLIP().unload_clip(
                FakeCLIP(patcher), object(), clip_module.MODE_CPU
            )


if __name__ == "__main__":
    unittest.main()
