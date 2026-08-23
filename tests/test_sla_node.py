"""Tests for the dependency-light ComfyUI SLA node wrapper."""

from __future__ import annotations

import importlib.util
from pathlib import Path
import sys
import types
import unittest


_ROOT = Path(__file__).resolve().parents[1]


def _load_node_module():
    package_name = "ddht_sla_node_test_package"
    package = types.ModuleType(package_name)
    package.__path__ = [str(_ROOT)]
    sys.modules[package_name] = package

    spec = importlib.util.spec_from_file_location(
        f"{package_name}.sla_attention",
        _ROOT / "sla_attention.py",
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


node_module = _load_node_module()


class FakeModel:
    pass


class SLANodeTests(unittest.TestCase):
    def test_node_registration_and_schema(self):
        node_class = node_module.NODE_CLASS_MAPPINGS["DDHTH3SLAAttention"]
        required = node_class.INPUT_TYPES()["required"]
        self.assertEqual(node_class.RETURN_TYPES, ("MODEL",))
        self.assertEqual(required["block_size"][0], ["64", "128"])
        self.assertEqual(required["sparsity_ratio"][1]["default"], 0.90)
        self.assertIn("protect_audio", required)

    def test_disabled_node_is_identity(self):
        model = FakeModel()
        node = node_module.DDHTH3SLAAttention()
        result = node.apply_sla(model, enabled=False)
        self.assertIs(result[0], model)

    def test_missing_backend_falls_back_to_original_model(self):
        model = FakeModel()
        node = node_module.DDHTH3SLAAttention()
        # The isolated test package has no ddht_sla submodule, emulating a
        # missing Triton/backend import. The node must remain a safe no-op.
        result = node.apply_sla(model, enabled=True)
        self.assertIs(result[0], model)


if __name__ == "__main__":
    unittest.main()
