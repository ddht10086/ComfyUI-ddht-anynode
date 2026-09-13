"""Pure-Python tests; no ComfyUI, Torch, GPU, or LLM server required."""

import importlib.util
from pathlib import Path
import sys
import types
import unittest
from unittest.mock import Mock, patch


ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("ddht_prefix_test", ROOT / "text_validation.py")
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
Node = module.DDHTTextPrefixGate


class TextPrefixGateTests(unittest.TestCase):
    def setUp(self):
        self.node = Node()

    def test_schema_and_registration(self):
        self.assertIs(module.NODE_CLASS_MAPPINGS["DDHT_TextPrefixGate"], Node)
        required = Node.INPUT_TYPES()["required"]
        self.assertTrue(required["text"][1]["forceInput"])
        self.assertTrue(required["expected_prefix"][1]["multiline"])
        self.assertFalse(required["ignore_leading_whitespace"][1]["default"])
        self.assertEqual(Node.RETURN_TYPES, ("STRING",))
        self.assertTrue(callable(getattr(self.node, Node.FUNCTION)))

    def test_matching_chinese_prefix_returns_original_object(self):
        text = "视频提示词：人物站在房间里。\n下一段。  "
        self.assertIs(self.node.check_prefix(text, "视频提示词：")[0], text)

    def test_exact_prefix_alone_passes(self):
        self.assertEqual(self.node.check_prefix("正确", "正确"), ("正确",))

    def test_middle_occurrence_does_not_pass(self):
        with self.assertRaisesRegex(RuntimeError, "已终止当前工作流"):
            self.node.check_prefix("下面是视频提示词：正文", "视频提示词：")

    def test_short_input_fails(self):
        with self.assertRaises(RuntimeError):
            self.node.check_prefix("视频", "视频提示词：")

    def test_empty_text_fails(self):
        with self.assertRaises(RuntimeError):
            self.node.check_prefix("", "正确")

    def test_whitespace_only_text_fails(self):
        with self.assertRaises(RuntimeError):
            self.node.check_prefix(" \n\t", "正确", True)

    def test_strict_mode_does_not_skip_whitespace(self):
        for whitespace in (" ", "\n", "\r\n", "\t", "\u3000"):
            with self.subTest(whitespace=whitespace), self.assertRaises(RuntimeError):
                self.node.check_prefix(whitespace + "正确：正文", "正确：")

    def test_ignore_whitespace_preserves_output(self):
        text = " \r\n\t\u3000正确：正文\n  "
        self.assertIs(self.node.check_prefix(text, "正确：", True)[0], text)

    def test_prefix_is_not_stripped(self):
        self.assertEqual(self.node.check_prefix(" 正文", " "), (" 正文",))
        with self.assertRaises(RuntimeError):
            self.node.check_prefix(" 正文", " 正", True)

    def test_case_sensitive(self):
        with self.assertRaises(RuntimeError):
            self.node.check_prefix("ok: text", "OK:")

    def test_fullwidth_punctuation_is_distinct(self):
        with self.assertRaises(RuntimeError):
            self.node.check_prefix("视频提示词:正文", "视频提示词：")

    def test_multiline_prefix(self):
        text = "结果：\n正确\n正文"
        self.assertEqual(self.node.check_prefix(text, "结果：\n正确"), (text,))

    def test_literal_backslash_n_is_not_a_newline(self):
        with self.assertRaises(RuntimeError):
            self.node.check_prefix("结果：\n正确", r"结果：\n正确")

    def test_literal_regex_characters(self):
        self.assertEqual(self.node.check_prefix("[OK].*正文", "[OK].*"), ("[OK].*正文",))
        with self.assertRaises(RuntimeError):
            self.node.check_prefix("OK正文", "^OK")

    def test_emoji_prefix(self):
        self.assertEqual(self.node.check_prefix("✅正确：正文", "✅正确："), ("✅正确：正文",))

    def test_no_unicode_normalization(self):
        with self.assertRaises(RuntimeError):
            self.node.check_prefix("e\u0301:正文", "é:")

    def test_bom_zero_width_thinking_and_fences_are_not_skipped(self):
        for extra in ("\ufeff", "\u200b", "<think>思考</think>", "```\n"):
            with self.subTest(extra=extra), self.assertRaises(RuntimeError):
                self.node.check_prefix(extra + "正确：正文", "正确：", True)

    def test_empty_prefix_rejected_before_execution(self):
        self.assertIsInstance(Node.VALIDATE_INPUTS(expected_prefix=""), str)
        self.assertIs(Node.VALIDATE_INPUTS(expected_prefix="正确："), True)
        self.assertIs(Node.VALIDATE_INPUTS(), True)  # Linked input checked at runtime.

    def test_empty_prefix_rejected_at_runtime(self):
        with self.assertRaisesRegex(ValueError, "不允许为空"):
            self.node.check_prefix("任意文本", "")

    def test_non_string_inputs_are_not_coerced(self):
        for text in (None, 12, ["正确"], {"text": "正确"}):
            with self.subTest(text=text), self.assertRaises(TypeError):
                self.node.check_prefix(text, "正确")
        for prefix in (None, 12, ["正确"]):
            with self.subTest(prefix=prefix), self.assertRaises(ValueError):
                self.node.check_prefix("正确", prefix)
        self.assertIsInstance(Node.VALIDATE_INPUTS(expected_prefix=12), str)

    def test_non_boolean_option_rejected(self):
        with self.assertRaises(TypeError):
            self.node.check_prefix("正确", "正确", "false")

    def test_mismatch_diagnostic_is_bounded_and_escaped(self):
        with self.assertRaises(RuntimeError) as caught:
            self.node.check_prefix("错误\n" + "x" * 10000, "正确" * 10000)
        message = str(caught.exception)
        self.assertIn("预期开头", message)
        self.assertIn("实际开头", message)
        self.assertIn(r"\n", message)
        self.assertNotIn("\n", message)
        self.assertLess(len(message), 400)

    def test_failed_gate_never_returns_bad_text_to_consumer(self):
        consumer = Mock()
        with self.assertRaises(RuntimeError):
            consumer(self.node.check_prefix("错误", "正确")[0])
        consumer.assert_not_called()

    def test_no_stale_state_after_success_or_failure(self):
        self.node.check_prefix("正确：第一段", "正确：")
        with self.assertRaises(RuntimeError):
            self.node.check_prefix("错误", "正确：")
        self.assertEqual(self.node.check_prefix("正确：第二段", "正确："), ("正确：第二段",))
        with self.assertRaises(RuntimeError):
            self.node.check_prefix("正确：第二段", "其他：")

    def test_package_entry_registers_node_and_preserves_other_mappings(self):
        package_name = "ddht_prefix_package_test"
        stubs = {}
        for name in ("clip_management", "nodes", "local_llm", "sla_attention"):
            stub = types.ModuleType(f"{package_name}.{name}")
            stub.NODE_CLASS_MAPPINGS = {name: object()}
            stub.NODE_DISPLAY_NAME_MAPPINGS = {name: name}
            stubs[stub.__name__] = stub
        stubs[f"{package_name}.text_validation"] = module
        package_spec = importlib.util.spec_from_file_location(
            package_name, ROOT / "__init__.py", submodule_search_locations=[str(ROOT)]
        )
        package = importlib.util.module_from_spec(package_spec)
        stubs[package_name] = package
        with patch.dict(sys.modules, stubs):
            package_spec.loader.exec_module(package)
        self.assertIs(package.NODE_CLASS_MAPPINGS["DDHT_TextPrefixGate"], Node)
        self.assertEqual(len(package.NODE_CLASS_MAPPINGS), 5)
        self.assertEqual(package.WEB_DIRECTORY, "./web")


if __name__ == "__main__":
    unittest.main()
