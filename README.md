# ComfyUI-ddht-anynode

一组专注、实用的 ComfyUI 自定义节点。

## 当前节点

### 保存视频（仅一个文件）- DDHT

把 `IMAGE` 图片序列编码为 MP4，可选连接 ComfyUI `AUDIO`。

- `frame_rate` 使用整数输入。
- 有音频时只保存最终的有声 MP4。
- 无音频时只保存最终的无声 MP4。
- 不额外保存首帧 PNG。
- 不生成或保留无声中间视频。
- 输出格式与 `pix_fmt` 对齐 VideoHelperSuite：支持 `video/h264-mp4` 与 `video/h265-mp4`；切换格式时会使用对应的像素格式顺序和 CRF 默认值。
- 音频编码为 AAC；短音频会补静音，长音频会裁切到视频长度。
- 兼容普通 ComfyUI `AUDIO` 字典和 VideoHelperSuite Load Video 输出的惰性音频对象；连接了无效音频时会明确报错，不再静默保存无声文件。

`quality` 是 FFmpeg CRF：数值越低质量越高、文件越大。与 VideoHelperSuite 一致，H.264 默认 `19`，H.265 默认 `22`。

### 按每秒帧数抽帧 - DDHT

输入一个 ComfyUI `IMAGE` 批次，根据原始帧率和目标抽帧率均匀输出较小的图片序列，适合提交给视觉 LLM 分析视频。

- `source_fps`：输入图片序列的实际帧率。
- `extract_fps`：每秒希望保留多少帧，使用整数输入。
- 输出抽取后的 `IMAGE` 批次和实际帧数。
- 当目标帧率不低于原始帧率时，直接返回全部帧，不生成重复帧。

如果图片来自 VideoHelperSuite 的 Load Video，请把其 `loaded_fps` 连接到本节点的 `source_fps`。

### 文本长度门控 - DDHT

检查输入文本的字符长度，只允许合理长度的 LLM 输出继续传递到下游节点。

- `text`：需要检查的文本，可连接其他节点的 `STRING` 输出。
- `min_length`：允许的最小字符数。
- `max_length`：允许的最大字符数。
- 范围包含最小值和最大值。
- 合格时原样输出文本，同时输出实际字符数。
- 过短或过长时使用 ComfyUI 的 `ExecutionBlocker` 停止全部下游执行。
- 如果旧版 ComfyUI 不支持 `ExecutionBlocker`，会抛出明确错误并终止当前工作流。

字符数使用 Python Unicode 字符计数，空格、换行和标点也会计入长度。

### 局域网多模态 LLM 推理 - DDHT

通过 HTTP 调用局域网中已经启动的本地大模型服务。ComfyUI 只负责整理提示词、压缩图片、发送请求和接收流式结果，不在节点进程中重复加载模型。

- 支持 `llama.cpp`、`Ollama`、`vLLM`、`SGLang` 和通用 OpenAI 兼容接口。
- 支持纯文本推理，也支持最多 8 个 `IMAGE` 输入端口。
- 初始只显示一个图片端口；连接后自动增加下一个端口，最多 8 个。
- 每个图片端口都可以接收完整批次，节点会按端口顺序展开。
- 图片总数超过 `最大图片数` 时，会在全部图片中均匀抽取，默认最多发送 24 张。
- 图片仅在内存中缩放并压缩为 JPEG，不创建临时图片文件。
- 支持 SSE（OpenAI 兼容）和 NDJSON（Ollama）流式输出，可响应 ComfyUI 的中断操作。
- 达到字符上限时可选择立即报错终止，或断开生成连接后返回截断文本。
- 输出生成文本、返回字符数，以及包含 token 用量、图片数和结束原因的 JSON。

常用地址示例：

| 框架 | `API地址` 示例 | 节点实际调用 |
|---|---|---|
| llama.cpp | `http://192.168.1.20:8080` | `/v1/chat/completions` |
| Ollama | `http://192.168.1.20:11434` | `/api/chat` |
| vLLM | `http://192.168.1.20:8000` | `/v1/chat/completions` |
| SGLang | `http://192.168.1.20:30000` | `/v1/chat/completions` |

也可以直接填写完整的 `/v1/chat/completions` 或 `/api/chat` 地址。使用 Ollama 时，`模型名称` 必须是服务中已经拉取的视觉模型名称。其他框架同样需要加载支持图片输入的模型，纯文本模型无法处理图片。

如服务需要密钥，请先在启动 ComfyUI 的系统环境中设置密钥，再把环境变量的名称填入 `API密钥环境变量`。不要直接把密钥写进工作流。`高级参数JSON` 可添加框架特有参数，但不能覆盖节点生成的 `messages` 和 `stream`。

## 安装

```bash
cd ComfyUI/custom_nodes
git clone https://github.com/ddht10086/ComfyUI-ddht-anynode.git
cd ComfyUI-ddht-anynode
pip install -r requirements.txt
```

安装后重启 ComfyUI。视频节点位于 `DDHT/Video`，文本门控位于 `DDHT/Text`，本地大模型节点位于 `DDHT/LLM`。

## 说明

保存视频需要 FFmpeg。节点会优先使用系统 PATH 中的 FFmpeg，否则使用 `imageio-ffmpeg` 提供的可执行文件。

本项目的功能设计参考了 [ComfyUI-VideoHelperSuite](https://github.com/Kosinkadink/ComfyUI-VideoHelperSuite)，并针对单文件输出和按时间均匀抽帧进行了独立实现。

## License

GNU General Public License v3.0。详见 [LICENSE](LICENSE)。
