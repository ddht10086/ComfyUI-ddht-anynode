# DeepSeek API 推理 - DDHT

分类 `DDHT/LLM`，节点类型 `DDHT_DeepSeekAPI`。

发送提示词、系统提示词和可选图片到 DeepSeek 官方云端 API，输出回答。无需本地加载 DeepSeek 模型，已有 `requests` 依赖即可使用。每次排队执行都会重新请求并按官方规则计费；不会自动重试。

## 使用方法

1. 更新节点包（`git pull`），重启 ComfyUI，并刷新浏览器前端。
2. 添加“DeepSeek API 推理 - DDHT”。填写或连接 `提示词`，可选填写 `系统提示词`。
3. 选择 `模型`；图片任务默认选择 `deepseek-flash`。
4. 在 `API_Key` 填写自己的密钥，或留空并设置运行 ComfyUI 的环境变量 `DEEPSEEK_API_KEY`。
5. 可选连接 `图片1`～`图片8`。每连接一个端口会出现下一个，最多八个；每个端口可以接收 IMAGE 批次。
6. 将输出 `文本` 接到文本开头校验、文本长度门控或其他文本处理节点。

直接输入密钥优先于环境变量。直接填写的密钥可能被 ComfyUI 或其他插件保存在工作流、执行历史、错误输入日志、生成文件的工作流元数据中；分享这些内容前请检查。使用环境变量可以避免在节点输入中保存密钥。本节点不把密钥加入请求正文、用量 JSON 或自身错误消息；不会读取服务端错误正文。不要把实际密钥提交到代码仓库。

## 模型与请求

根据 2026-09-16 的 [官方模型说明](https://api-docs.deepseek.com/zh-cn/quick_start/pricing/)：

| 选择 | 图片 | 说明 |
| --- | --- | --- |
| `deepseek-flash` | 支持 | 默认，用于文本和多图任务 |
| `deepseek-v4-pro` | 不支持 | 纯文本；连接图片时会在请求前报错 |
| 自定义 | 取决于模型 | 填写官方模型 ID；服务端决定是否可用 |

固定调用 `https://api.deepseek.com/chat/completions`，使用 Bearer 认证和 SSE。保留 TLS 校验，禁止自动跳转，不会自动转发到第三方服务。

## 图片处理

按照图片端口编号、端口内批次顺序发送，并依次标记图1、图2等。复用本项目的 IMAGE 转换：等比例缩小、透明区域合成白底、内存中编码 JPEG/Base64，不产生图片临时文件。

- `最大图片数` 默认 24；超限报错，不会抽样或丢弃图片。视频帧可先经过“按每秒帧数抽帧”节点。
- `图片最大边长` 默认 1024，最多 4096；不放大小图。
- `JPEG质量` 默认 90。
- `图片细节` 默认 auto，可选择 low/high/original。
- 根据 [官方图片接口](https://api-docs.deepseek.com/guides/vision/)，内联请求正文上限为 48 MiB。本节点检查实际序列化后的大小，超限不发送。

## 生成设置与输出

`最大生成token` 默认 8192，`思考模式` 默认自动（由服务端使用默认值）。开启时可设置 low/high/max 推理强度；关闭时才发送温度参数，遵循 [官方思考模式说明](https://api-docs.deepseek.com/guides/thinking_mode/)。

| 输出 | 内容 |
| --- | --- |
| 文本 | 最终回答，不混入思考内容，可直接进行前缀校验 |
| 字符数 | 最终回答的 Unicode 字符数量 |
| 思考内容 | API 的 reasoning_content；未返回时为空字符串 |
| 用量JSON | 官方 token 统计、模型、结束原因、图片数、请求大小及耗时；不计算金额 |

`最大输出字符数` 默认 100000，统计回答和思考的总和。超限时断开连接并报错，不返回截断结果。token 上限导致的截断、空回答、网络断流或服务错误同样报错，避免把半段文本当成正确结果。

`生成超时秒` 默认 600。从 HTTP 请求开始计算，在读取事件时检查总时长；连接等待最多 10 秒，单次网络无数据等待最多 30 秒（均不超过配置超时）。网络阻塞时，总时长与取消检查可能延后至下一次数据到达或网络超时。取消时关闭响应连接；已被服务端处理的调用是否停止计费取决于 DeepSeek。

常见 HTTP 错误会给出中文提示：401 密钥错误、402 余额不足、429 限速，以及服务繁忙等，参见 [官方错误码](https://api-docs.deepseek.com/zh-cn/quick_start/error_codes/)。

## 验证

Python 测试覆盖请求构建、密钥读取、模拟 SSE、错误与取消、图片顺序和真实 JPEG 编码，以及本机模拟 HTTP 服务。图片测试中以 NumPy 适配对象替代 Torch 张量；实际 GPU/ComfyUI 工作流与真实付费 DeepSeek API 需在用户环境验证。

```bash
python -m unittest discover -s tests -p test_deepseek_api.py -v
node tests/test_dynamic_image_inputs.mjs
```
