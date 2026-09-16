"""Protocol, image conversion, node wiring, and loopback HTTP integration tests."""

import base64
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import importlib.util
import io
import json
import os
from pathlib import Path
import sys
import threading
import types
import unittest
from unittest.mock import Mock, patch

import numpy as np
from PIL import Image
import requests


ROOT = Path(__file__).resolve().parents[1]
PACKAGE = "ddht_deepseek_test"
package = types.ModuleType(PACKAGE)
package.__path__ = [str(ROOT)]
sys.modules[PACKAGE] = package


def load_module(name):
    spec = importlib.util.spec_from_file_location(f"{PACKAGE}.{name}", ROOT / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


client = load_module("deepseek_client")
node_module = load_module("deepseek_api")
TEST_KEY = "unit-test-credential"


def event(content=None, reasoning=None, finish=None):
    delta = {}
    if content is not None:
        delta["content"] = content
    if reasoning is not None:
        delta["reasoning_content"] = reasoning
    return {"id": "request-test", "model": "deepseek-flash", "choices": [{"index": 0, "delta": delta, "finish_reason": finish}]}


def wire_events(*events):
    return b"".join(("data: " + (value if isinstance(value, str) else json.dumps(value, ensure_ascii=False)) + "\n\n").encode() for value in events)


def successful_stream():
    return wire_events(
        event(reasoning="分析"), event("视频提示词："), event("描述图片。", finish="stop"),
        {"choices": [], "usage": {"prompt_tokens": 40, "completion_tokens": 10, "total_tokens": 50}},
        "[DONE]",
    )


class Response:
    def __init__(self, body=None, status=200):
        self.body = successful_stream() if body is None else body
        self.status_code = status
        self.closed = False

    def iter_lines(self, chunk_size):
        yield from self.body.splitlines()

    def close(self):
        self.closed = True


class Interrupted(BaseException):
    pass


class ClientTests(unittest.TestCase):
    def payload(self, **overrides):
        return client.build_payload(**{"model": "deepseek-flash", "prompt": "描述图片", **overrides})

    def run_response(self, response, **kwargs):
        with patch.object(requests, "post", return_value=response) as post:
            result = client.execute_request(self.payload(), TEST_KEY, **kwargs)
        return result, post

    def test_direct_and_environment_keys(self):
        with patch.dict(os.environ, {"DEEPSEEK_API_KEY": "env-key"}):
            self.assertEqual(client.resolve_api_key(" direct-key ", "DEEPSEEK_API_KEY"), "direct-key")
            self.assertEqual(client.resolve_api_key("", "DEEPSEEK_API_KEY"), "env-key")
        with patch.dict(os.environ, {}, clear=True), self.assertRaises(ValueError):
            client.resolve_api_key("", "DEEPSEEK_API_KEY")

    def test_invalid_key_error_does_not_echo_secret(self):
        with self.assertRaises(ValueError) as error:
            client.resolve_api_key("sensitive\nvalue", "")
        self.assertNotIn("sensitive", str(error.exception))

    def test_model_selection_and_vision_rejection(self):
        self.assertEqual(client.resolve_model("自定义", " future-model "), "future-model")
        with self.assertRaises(ValueError):
            client.resolve_model("自定义", "")
        with self.assertRaises(ValueError):
            self.payload(model="deepseek-v4-pro", images=["AA=="])
        self.payload(model="deepseek-v4-pro")
        self.payload(model="deepseek-v4-flash-vision-exp", images=["AA=="])

    def test_request_fields_and_thinking_modes(self):
        payload = self.payload(system_prompt="按格式回答", thinking="开启", reasoning_effort="max")
        self.assertEqual(payload["messages"][0], {"role": "system", "content": "按格式回答"})
        self.assertEqual(payload["thinking"], {"type": "enabled"})
        self.assertEqual(payload["reasoning_effort"], "max")
        self.assertNotIn("temperature", payload)
        plain = self.payload(thinking="关闭", reasoning_effort="high", temperature=0.3)
        self.assertEqual(plain["temperature"], 0.3)
        self.assertNotIn("reasoning_effort", plain)
        self.assertNotIn("thinking", self.payload())
        self.assertNotIn("temperature", self.payload())

    def test_images_are_ordered_in_user_message_only(self):
        payload = self.payload(images=["Zmlyc3Q=", "c2Vjb25k"], system_prompt="系统", image_detail="low")
        self.assertIsInstance(payload["messages"][0]["content"], str)
        content = payload["messages"][1]["content"]
        self.assertEqual([part["text"] for part in content if part["type"] == "text"], ["描述图片", "图1：", "图2："])
        self.assertEqual(content[2]["image_url"], {"url": "data:image/jpeg;base64,Zmlyc3Q=", "detail": "low"})

    def test_invalid_settings_and_size_rejected_before_network(self):
        for override in ({"prompt": " "}, {"max_tokens": 0}, {"temperature": float("nan")}, {"reasoning_effort": "wrong"}):
            with self.subTest(override=override), self.assertRaises(ValueError):
                self.payload(**override)
        with patch.object(client, "MAX_BODY_BYTES", 5), patch.object(requests, "post") as post:
            with self.assertRaises(ValueError):
                client.execute_request(self.payload(), TEST_KEY)
            post.assert_not_called()

    def test_complete_stream_text_reasoning_usage_and_transport(self):
        response = Response()
        result, post = self.run_response(response)
        self.assertEqual(result.text, "视频提示词：描述图片。")
        self.assertEqual(result.reasoning, "分析")
        self.assertEqual(result.usage["total_tokens"], 50)
        self.assertTrue(response.closed)
        self.assertEqual(post.call_count, 1)
        self.assertEqual(post.call_args.args[0], "https://api.deepseek.com/chat/completions")
        options = post.call_args.kwargs
        self.assertFalse(options["allow_redirects"])
        self.assertEqual(options["headers"]["Authorization"], f"Bearer {TEST_KEY}")
        self.assertNotIn(TEST_KEY.encode(), options["data"])
        self.assertTrue(json.loads(options["data"])["stream_options"]["include_usage"])

    def test_sse_comments_blank_lines_and_multiline_json(self):
        body = b": keepalive\n\nevent: message\n" + b"data: " + json.dumps(event("完成", finish="stop"), indent=2, ensure_ascii=False).encode().replace(b"\n", b"\ndata: ") + b"\n\ndata: [DONE]\n\n"
        result, _ = self.run_response(Response(body))
        self.assertEqual(result.text, "完成")

    def test_http_errors_are_actionable_and_do_not_echo_body(self):
        for code in (301, 400, 401, 402, 413, 422, 429, 500, 503):
            response = Response(TEST_KEY.encode(), status=code)
            with self.subTest(code=code), self.assertRaisesRegex(RuntimeError, f"HTTP {code}") as error:
                self.run_response(response)
            self.assertNotIn(TEST_KEY, str(error.exception))
            self.assertTrue(response.closed)

    def test_truncated_invalid_and_unfinished_streams_fail_closed(self):
        bodies = [
            wire_events(event("半句")), wire_events(event("半句"), "[DONE]"),
            wire_events(event("半句", finish="length"), "[DONE]"),
            wire_events(event("", finish="stop"), "[DONE]"),
            wire_events(event("拒绝", finish="content_filter"), "[DONE]"),
            wire_events({"error": {"message": TEST_KEY}}),
            b"data: not-json\n\n", b"data: []\n\n", b"data: \xff\n\n",
        ]
        for body in bodies:
            response = Response(body)
            with self.subTest(body=body[:50]), self.assertRaises(RuntimeError) as error:
                self.run_response(response)
            self.assertNotIn(TEST_KEY, str(error.exception))
            self.assertTrue(response.closed)

    def test_character_limit_covers_reasoning_and_closes_stream(self):
        response = Response(wire_events(event(reasoning="12345"), event("x", finish="stop"), "[DONE]"))
        with self.assertRaisesRegex(RuntimeError, "最大输出字符数"):
            self.run_response(response, max_output_characters=5)
        self.assertTrue(response.closed)
        result, _ = self.run_response(Response(wire_events(event("12345", finish="stop"), "[DONE]")), max_output_characters=5)
        self.assertEqual(result.text, "12345")

    def test_cancel_before_request_and_during_stream(self):
        with patch.object(requests, "post") as post, self.assertRaises(Interrupted):
            client.execute_request(self.payload(), TEST_KEY, check_interrupt=Mock(side_effect=Interrupted))
        post.assert_not_called()
        response = Response()
        check = Mock(side_effect=[None, None, Interrupted()])
        with self.assertRaises(Interrupted):
            self.run_response(response, check_interrupt=check)
        self.assertTrue(response.closed)

    def test_deadline_closes_response(self):
        response = Response()
        with patch.object(client.time, "monotonic", side_effect=[0, 0, 5]), self.assertRaisesRegex(RuntimeError, "生成超时"):
            self.run_response(response, timeout_seconds=1)
        self.assertTrue(response.closed)

    def test_network_errors_are_sanitized_and_never_retried(self):
        for error in (requests.Timeout(TEST_KEY), requests.ConnectionError(TEST_KEY)):
            with patch.object(requests, "post", side_effect=error) as post, self.assertRaises(RuntimeError) as caught:
                client.execute_request(self.payload(), TEST_KEY)
            self.assertNotIn(TEST_KEY, str(caught.exception))
            self.assertEqual(post.call_count, 1)

    def test_read_failure_closes_response(self):
        response = Response()
        response.iter_lines = Mock(side_effect=requests.ConnectionError(TEST_KEY))
        with self.assertRaises(RuntimeError):
            self.run_response(response)
        self.assertTrue(response.closed)


class ArrayTensor:
    """Exercise real JPEG/NumPy code with only Torch's tensor operations stubbed."""
    def __init__(self, array):
        self.array = np.asarray(array, dtype=np.float32)

    @property
    def shape(self):
        return self.array.shape

    @property
    def ndim(self):
        return self.array.ndim

    def unsqueeze(self, axis):
        return ArrayTensor(np.expand_dims(self.array, axis))

    def __iter__(self):
        return (ArrayTensor(frame) for frame in self.array)

    def detach(self):
        return self

    def to(self, **kwargs):
        return self

    def clamp(self, low, high):
        return ArrayTensor(np.clip(self.array, low, high))

    def numpy(self):
        return self.array


class NodeAndImageTests(unittest.TestCase):
    def setUp(self):
        comfy = types.ModuleType("comfy")
        mm = types.ModuleType("comfy.model_management")
        mm.processing_interrupted = lambda: False
        mm.InterruptProcessingException = Interrupted
        comfy.model_management = mm
        utils = types.ModuleType("comfy.utils")
        utils.ProgressBar = Mock(return_value=Mock())
        torch = types.ModuleType("torch")
        torch.Tensor, torch.float32 = ArrayTensor, object()
        self.modules = patch.dict(sys.modules, {
            "comfy": comfy, "comfy.model_management": mm, "comfy.utils": utils, "torch": torch,
        })
        self.modules.start()
        self.addCleanup(self.modules.stop)
        self.local = load_module("local_llm")

    def test_schema_model_choices_and_eight_inputs(self):
        node = node_module.DDHTDeepSeekAPI
        schema = node.INPUT_TYPES()
        self.assertEqual(schema["required"]["模型"][1]["default"], "deepseek-flash")
        self.assertEqual(sum(spec[0] == "IMAGE" for spec in schema["optional"].values()), 8)
        self.assertTrue(math_is_nan(node.IS_CHANGED()))
        self.assertEqual(node.RETURN_TYPES, ("STRING", "INT", "STRING", "STRING"))

    def test_real_jpeg_resize_alpha_and_non_mutation(self):
        original = np.zeros((1, 120, 240, 4), dtype=np.float32)
        batch = ArrayTensor(original)
        images, ports = node_module.encode_images({"图片1": batch}, "deepseek-flash", 24, 64, 90, lambda: None)
        self.assertEqual(ports, 1)
        image = Image.open(io.BytesIO(base64.b64decode(images[0])))
        self.assertEqual(image.format, "JPEG")
        self.assertEqual(image.size, (64, 32))
        self.assertTrue(all(v > 245 for v in image.getpixel((20, 20))))
        np.testing.assert_array_equal(batch.array, original)

    def test_port_and_batch_order_with_sparse_ports(self):
        black = ArrayTensor(np.zeros((2, 8, 8, 3)))
        white = ArrayTensor(np.ones((8, 8, 3)))
        images, ports = node_module.encode_images({"图片8": white, "图片2": black}, "deepseek-flash", 3, 64, 90, lambda: None)
        self.assertEqual((len(images), ports), (3, 2))
        colors = [Image.open(io.BytesIO(base64.b64decode(value))).getpixel((0, 0))[0] for value in images]
        self.assertEqual(colors, [0, 0, 255])

    def test_over_limit_never_encodes_or_drops_frames(self):
        with patch.object(self.local, "_frame_to_jpeg_base64") as encode, self.assertRaisesRegex(ValueError, "不会自动丢弃"):
            node_module.encode_images({"图片1": ArrayTensor(np.zeros((3, 8, 8, 3)))}, "deepseek-flash", 2, 64, 90, lambda: None)
        encode.assert_not_called()

    def test_empty_invalid_and_unsupported_images(self):
        for value in ("wrong", ArrayTensor(np.zeros((0, 8, 8, 3)))):
            with self.subTest(value=type(value).__name__), self.assertRaises(ValueError):
                node_module.encode_images({"图片1": value}, "deepseek-flash", 24, 64, 90, lambda: None)
        with self.assertRaisesRegex(ValueError, "不支持图片"):
            node_module.encode_images({"图片1": object()}, "deepseek-v4-pro", 24, 64, 90, lambda: None)

    def test_infer_outputs_are_separate_and_work_with_prefix_gate(self):
        response = Response()
        images = ArrayTensor(np.ones((1, 8, 8, 3)))
        with patch.object(requests, "post", return_value=response) as post:
            text, count, reasoning, info = node_module.DDHTDeepSeekAPI().infer(
                "按要求写视频提示词", API_Key=TEST_KEY, 图片1=images,
            )
        self.assertEqual(count, len(text))
        self.assertEqual(reasoning, "分析")
        self.assertNotIn("<think>", text)
        self.assertEqual(json.loads(info)["sent_image_count"], 1)
        self.assertNotIn(TEST_KEY, info)
        self.assertIn("image_url", post.call_args.kwargs["data"].decode())
        gate = load_module("text_validation").DDHTTextPrefixGate()
        self.assertEqual(gate.check_prefix(text, "视频提示词："), (text,))


def math_is_nan(value):
    return value != value


class LoopbackHTTPTests(unittest.TestCase):
    def test_real_requests_sse_roundtrip_with_images(self):
        received = {}

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *args):
                pass

            def do_POST(self):
                body = self.rfile.read(int(self.headers["Content-Length"]))
                received.update(payload=json.loads(body), auth=self.headers.get("Authorization"), path=self.path)
                self.send_response(200)
                self.send_header("Content-Type", "text/event-stream")
                self.end_headers()
                body = successful_stream()
                for position in range(0, len(body), 7):
                    self.wfile.write(body[position:position + 7])
                self.wfile.flush()

        server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            url = f"http://127.0.0.1:{server.server_port}/chat/completions"
            payload = client.build_payload(model="deepseek-flash", prompt="看图", images=["AA=="])
            with patch.object(client, "ENDPOINT", url):
                result = client.execute_request(payload, TEST_KEY)
            self.assertEqual(result.text, "视频提示词：描述图片。")
            self.assertEqual(received["path"], "/chat/completions")
            self.assertEqual(received["auth"], f"Bearer {TEST_KEY}")
            self.assertEqual(received["payload"]["messages"][0]["content"][2]["type"], "image_url")
        finally:
            server.shutdown()
            server.server_close()
            thread.join(timeout=2)


if __name__ == "__main__":
    unittest.main()
