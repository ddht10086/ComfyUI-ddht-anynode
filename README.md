# ComfyUI-ddht-anynode

一组专注、实用的 ComfyUI 自定义节点。

## 当前节点

### 保存视频（仅一个文件）- DDHT

把 `IMAGE` 图片序列编码为 MP4，可选连接 ComfyUI `AUDIO`。

- 有音频时只保存最终的有声 MP4。
- 无音频时只保存最终的无声 MP4。
- 不额外保存首帧 PNG。
- 不生成或保留无声中间视频。
- 支持 H.264 与 H.265，默认 H.264。
- 音频编码为 AAC；短音频会补静音，长音频会裁切到视频长度。

`quality` 是 FFmpeg CRF：数值越低质量越高、文件越大，默认 `18`。

### 按每秒帧数抽帧 - DDHT

输入一个 ComfyUI `IMAGE` 批次，根据原始帧率和目标抽帧率均匀输出较小的图片序列，适合提交给视觉 LLM 分析视频。

- `source_fps`：输入图片序列的实际帧率。
- `extract_fps`：每秒希望保留多少帧。
- 输出抽取后的 `IMAGE` 批次和实际帧数。
- 当目标帧率不低于原始帧率时，直接返回全部帧，不生成重复帧。

如果图片来自 VideoHelperSuite 的 Load Video，请把其 `loaded_fps` 连接到本节点的 `source_fps`。

## 安装

```bash
cd ComfyUI/custom_nodes
git clone https://github.com/ddht10086/ComfyUI-ddht-anynode.git
cd ComfyUI-ddht-anynode
pip install -r requirements.txt
```

安装后重启 ComfyUI，在 `DDHT/Video` 分类中寻找节点。

## 说明

保存视频需要 FFmpeg。节点会优先使用系统 PATH 中的 FFmpeg，否则使用 `imageio-ffmpeg` 提供的可执行文件。

本项目的功能设计参考了 [ComfyUI-VideoHelperSuite](https://github.com/Kosinkadink/ComfyUI-VideoHelperSuite)，并针对单文件输出和按时间均匀抽帧进行了独立实现。

## License

GNU General Public License v3.0。详见 [LICENSE](LICENSE)。
