"""Dependency-free regression tests for the DDHT SLA wrapper chain."""

from __future__ import annotations

import importlib.util
from pathlib import Path
import sys
import types
import unittest


_ROOT = Path(__file__).resolve().parents[1]
_PATCH = _ROOT / "ddht_sla" / "patch.py"
_PACKAGE = "ddht_sla_wrapper_test"


def _load_patch_module():
    torch_stub = types.ModuleType("torch")
    torch_stub.bfloat16 = object()
    torch_stub.float16 = object()
    sys.modules.setdefault("torch", torch_stub)

    package = types.ModuleType(_PACKAGE)
    package.__path__ = [str(_ROOT)]
    sys.modules[_PACKAGE] = package

    sla_package = types.ModuleType(f"{_PACKAGE}.ddht_sla")
    sla_package.__path__ = [str(_ROOT / "ddht_sla")]
    sys.modules[sla_package.__name__] = sla_package

    block_map = types.ModuleType(f"{_PACKAGE}.ddht_sla.block_map")
    block_map.get_block_map = lambda *args, **kwargs: (_ for _ in ()).throw(
        AssertionError("routing is outside this wrapper test")
    )
    sys.modules[block_map.__name__] = block_map

    kernel = types.ModuleType(f"{_PACKAGE}.ddht_sla.kernel")
    kernel.block_sparse_attention = lambda *args, **kwargs: (_ for _ in ()).throw(
        AssertionError("the Triton kernel is outside this wrapper test")
    )
    sys.modules[kernel.__name__] = kernel

    spec = importlib.util.spec_from_file_location(
        f"{_PACKAGE}.ddht_sla.patch",
        _PATCH,
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


sla_patch = _load_patch_module()


class WrapperExecutor:
    """Small behavioral model of ComfyUI's current WrapperExecutor."""

    def __init__(self, original, wrappers, index=0):
        self.original = original
        self.wrappers = list(wrappers)
        self.index = index
        self.is_last = index == len(self.wrappers)

    def __call__(self, *args, **kwargs):
        return WrapperExecutor(
            self.original,
            self.wrappers,
            self.index + 1,
        ).execute(*args, **kwargs)

    def execute(self, *args, **kwargs):
        if self.is_last:
            return self.original(*args, **kwargs)
        return self.wrappers[self.index](self, *args, **kwargs)


class SLAWrapperChainTests(unittest.TestCase):
    def test_patch_clones_model_and_installs_both_hooks(self):
        class FakeModel:
            def __init__(self):
                self.model_options = {"transformer_options": {"existing": 1}}
                self.wrapper = None

            def clone(self):
                return FakeModel()

            def add_wrapper_with_key(self, wrapper_type, key, wrapper):
                self.wrapper = (wrapper_type, key, wrapper)

        original = FakeModel()
        patched = sla_patch.patch_h3_sla(original)

        self.assertIsNot(patched, original)
        options = patched.model_options["transformer_options"]
        self.assertEqual(options["existing"], 1)
        self.assertTrue(callable(options["optimized_attention_override"]))
        self.assertEqual(patched.wrapper[:2], ("diffusion_model", "ddht_h3_sla_state"))
        self.assertTrue(callable(patched.wrapper[2]))

    def test_sla_advances_to_later_wrappers(self):
        events = []
        state = sla_patch._new_state()
        sla_wrapper = sla_patch._make_wrapper(state, 0.90, 64, 64, 0)

        def downstream(executor, *args, **kwargs):
            events.append("downstream")
            return executor(*args, **kwargs)

        def original(*args, **kwargs):
            events.append("original")
            return "ok"

        executor = WrapperExecutor(original, [sla_wrapper, downstream])
        result = executor.execute(
            object(),
            object(),
            object(),
            transformer_options={"sample_sigmas": [1.0, 0.0]},
        )

        self.assertEqual(result, "ok")
        self.assertEqual(events, ["downstream", "original"])

    def test_h3_payload_and_protected_prefix_are_forwarded(self):
        seen = {}
        state = sla_patch._new_state()
        sla_wrapper = sla_patch._make_wrapper(state, 0.90, 64, 64, 0)

        class Layout:
            segments = [(0, 10, "text"), (10, 40, "audio"), (40, 100, "video")]

        payload = {"layout": Layout()}

        def downstream(executor, *args, **kwargs):
            seen.update(kwargs)
            return executor(*args, **kwargs)

        executor = WrapperExecutor(lambda *args, **kwargs: None, [sla_wrapper, downstream])
        executor.execute(
            object(),
            object(),
            object(),
            transformer_options={"sample_sigmas": [1.0, 0.0]},
            minimax_payload=payload,
        )

        self.assertIs(seen["minimax_payload"], payload)
        self.assertEqual(
            seen["transformer_options"]["_ddht_h3sla_prefix"],
            40,
        )

    def test_non_h3_call_does_not_receive_minimax_payload(self):
        seen = {}
        state = sla_patch._new_state()
        sla_wrapper = sla_patch._make_wrapper(state, 0.90, 64, 64, 0)

        def downstream(executor, *args, **kwargs):
            seen.update(kwargs)
            return executor(*args, **kwargs)

        executor = WrapperExecutor(lambda *args, **kwargs: None, [sla_wrapper, downstream])
        executor.execute(
            object(),
            object(),
            object(),
            transformer_options={"sample_sigmas": [1.0, 0.0]},
        )

        self.assertNotIn("minimax_payload", seen)

    def test_dense_last_steps_uses_logical_sampler_position(self):
        state = sla_patch._new_state()
        sla_wrapper = sla_patch._make_wrapper(state, 0.90, 64, 64, 2)
        dense_flags = []

        def downstream(executor, *args, **kwargs):
            dense_flags.append(
                kwargs["transformer_options"]["_ddht_h3sla_dense"]
            )
            return executor(*args, **kwargs)

        executor = WrapperExecutor(lambda *args, **kwargs: None, [sla_wrapper, downstream])
        schedule = [float(19 - index) for index in range(20)]

        for index in [0, 5, 17, 18]:
            executor.execute(
                object(),
                object(),
                object(),
                transformer_options={
                    "sample_sigmas": schedule,
                    "sigmas": [schedule[index]],
                },
            )

        self.assertEqual(dense_flags, [False, False, True, True])

    def test_non_callable_legacy_executor_still_works(self):
        seen = []

        class LegacyExecutor:
            @staticmethod
            def original(*args, **kwargs):
                seen.append("original")
                return "ok"

        result = sla_patch._call_next_wrapper(LegacyExecutor, 1, 2, three=3)
        self.assertEqual(result, "ok")
        self.assertEqual(seen, ["original"])


if __name__ == "__main__":
    unittest.main()
