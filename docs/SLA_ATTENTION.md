# MiniMax-H3 SLA Attention - DDHT

该节点为 MiniMax-H3 安装 SLA 块稀疏自注意力，用于配合 lightx2v 的
SLA turbo LoRA。节点只修改克隆后的 `MODEL`，不会改写模型权重。

## 使用方法

连接顺序：模型 → LoRA 加载器 → `MiniMax-H3 SLA Attention - DDHT` →
采样器。本节点应放在所有 LoRA 加载器之后。

- `sparsity_ratio`：跳过的键块比例；`0.85` 是 lightx2v 的原始设置，
  `0.90` 更快但更稀疏。
- `block_size`：建议有声视频使用 `64`；`128` 略快，但可能降低语音质量。
- `min_seq_len`：短于此长度自动使用原密集注意力，避免短序列反而变慢。
- `dense_last_steps`：让最后 N 个采样步骤恢复密集注意力，以尝试保留细节。
- `protect_audio`：固定保留文本、条件和音频前缀；有声视频建议开启。
- `enabled`：关闭后原样传递模型，便于速度对照。

节点分类为 `DDHT/Model Patches/MiniMax-H3`。

## 适用范围与依赖

该功能不是通用模型加速器，仅针对 MiniMax-H3 的注意力布局。它需要：

- 可用的 Triton；
- 兼容的 NVIDIA GPU；
- 支持 `optimized_attention_override` 和 `add_wrapper_with_key` 的较新
  ComfyUI。

非 H3 注意力、短序列或内核执行失败时会安全回退到原密集注意力。
SLA turbo LoRA 需要单独加载，本节点不会下载或加载 LoRA。

## 来源与许可

功能移植并适配自
[PlagueKind/ComfyUI-PlagueKind-Nodes](https://github.com/PlagueKind/ComfyUI-PlagueKind-Nodes/tree/c5b14d730cff13325b1836915bcd63b1506fbff3/ComfyUI-H3-SLA-Attention)，
其块选择和 Triton 内核又源自
[ModelTC/LightX2V](https://github.com/ModelTC/LightX2V)。第三方版权与许可证
副本见仓库根目录的 `THIRD_PARTY_NOTICES.md` 和 `LICENSES/`。
