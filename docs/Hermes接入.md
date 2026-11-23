# 在 Hermes Agent 中使用本机 Drama 3 教师服务（8193）

如使用独立 NVFP4 实时服务，同机地址为 `http://127.0.0.1:8195/v1`。跨机请使用实际配置的服务器 IP。实时服务的启动命令、人声 ID、语气控制和音频格式见[8195 实时 API 操作手册](中英实时语音.md#hermes-配置)。本页以下命令适用于 8193 教师服务。

本机已有 Windows 版 Hermes Agent 0.21.2，安装提交为 `b6b53c69a6ed49cb099cf1bfe76b5e6edd718e5a`，内置 OpenAI SDK 2.24.0。Hermes 自带的语音供应商可以把 `text_to_speech` 工具发到 Drama 3。本方案只在一个单独的 Hermes 配置里设置语音，不需要登录云端、配置聊天模型或安装额外的 Hermes 插件。[Hermes 官方语音文档](https://hermes-agent.nousresearch.com/docs/user-guide/features/tts) · [对应版本的语音路由代码](https://github.com/NousResearch/hermes-agent/blob/b6b53c69a6ed49cb099cf1bfe76b5e6edd718e5a/tools/tts_tool_openai.py)

## 先生成一个文件

本机服务监听 `127.0.0.1:8193`。在仓库根目录打开 PowerShell，运行下面两行。脚本创建临时 Hermes 配置，以真实工具名 `text_to_speech` 调用注册表，再把完成的 WAV 存到给定位置。输出文件名需要是新文件。

```powershell
$hermesPython = Join-Path $env:LOCALAPPDATA 'hermes\hermes-agent\venv\Scripts\python.exe'
& $hermesPython -B -X utf8 integrations/hermes/speak.py '你好，我是你的本地语音助手。' 'outputs/hermes-demo.wav'
```

把扩展名改成 `.mp3` 或 `.ogg` 可以生成 MP3 或 Ogg/Opus。WAV 保留教师的 44,100 Hz 单声道音频。`--instructions '用放松温和的语气朗读。'` 可给整文件合成传导演描述。添加 `--workspace-profile` 可使用工作区专用配置中选定的声音。`.pcm` 会走 Hermes 原生语音流接口，输出为 24,000 Hz、单声道、有符号 16 位小端裸 PCM，可使用 `ffplay -f s16le -ar 24000 -ch_layout mono outputs/hermes-demo.pcm` 播放。[独立调用脚本](../integrations/hermes/speak.py)

## 为某个 Hermes 会话启用语音

使用这份独立配置的会话会把语音请求发到 `http://127.0.0.1:8193/v1/audio/speech`。`base_url` 填到 `/v1` 即可，Hermes 会补上 `/audio/speech`。配置中的 `openai` 指 Hermes 内置供应商名称，`drama-3` 指实际运行模型。本机默认角色是 `default`，也可以在 `voice` 填入 `GET /v1/audio/voices` 返回的设计别名或已保存的角色 ID。配置内的固定占位字符串满足 Hermes SDK 对该字段的要求。[配置模板](../integrations/hermes/config.yaml) · [Hermes 配置说明](https://hermes-agent.nousresearch.com/docs/user-guide/configuration)

可以在工作区创建专用 profile。在**新的 PowerShell 窗口**从仓库根目录执行下面两行即可启动。关闭这个终端后，其他窗口与桌面程序继续使用它们原来的配置。[可复制的配置模板](../integrations/hermes/config.yaml)

```powershell
New-Item -ItemType Directory -Force '.runtime/hermes-drama3' | Out-Null
if (!(Test-Path '.runtime/hermes-drama3/config.yaml')) { Copy-Item 'integrations/hermes/config.yaml' '.runtime/hermes-drama3/config.yaml' }
$env:HERMES_HOME = (Resolve-Path '.runtime\hermes-drama3').Path
hermes
```

若该专用会话已配置你要使用的聊天模型并启用了 `tts` 工具集，可以让它「用 text_to_speech 把这句话保存成语音」。对已有的对话，也可通过 Hermes 的 `/voice tts` 打开语音回复。命令行单独生成文件无需聊天模型。麦克风识别、VAD、电话网关、对话模型和通话路由由 Hermes 或其他宿主配置。[官方语音模式说明](https://hermes-agent.nousresearch.com/docs/user-guide/features/voice-mode)

## 版本与验收

**已跑通的真实路径。** 工作区专用 profile 通过 Hermes 的 `text_to_speech` 注册工具，将 “Welcome home.” 发给 8193 的语音服务，由 8194 的秋叶工作进程执行 FP8 教师。Hermes 在 68.42 秒后收到 WAV，音长 1.161 秒，44,100 Hz 单声道，共 51,200 个样本。独立在 CPU 上运行的多语种 Whisper small 回读 “Welcome home.”，英文词错率为 0。该时间包含请求排队、加载、生成和交付中的实际耗时。试听样音（`outputs/provider-validation/hermes/welcome.wav`） · Hermes 收据及教师报告（`outputs/provider-validation/hermes/receipt.json`） · 独立回读（`outputs/provider-validation/hermes/asr.json`）

本机 Windows 安装位置为 `%LOCALAPPDATA%\hermes\hermes-agent`。Ubuntu-22.04 的命令搜索路径和常见安装目录未发现 Hermes。Ubuntu 请用安装 Hermes 的环境执行 `/实际路径/bin/python integrations/hermes/speak.py "你好" outputs/hermes-demo.wav`，并确保 8193 位于同一网络环境。

当前安装版的文件工具按扩展名发送 `response_format`，其中 `.ogg` 对应 `opus`。语音模式另发 `pcm` 并按固定 24 kHz 解码。当前服务会在本次文本合成完毕后发送整段音频。Hermes 可以逐句请求；同一句内的首个音频包要等该句生成完毕。文件返回和 PCM 数据消费可在无声卡条件下测试，扬声器、桌面应用、电话和第三方消息平台应分别验听。[本机对应的流播放实现](https://github.com/NousResearch/hermes-agent/blob/b6b53c69a6ed49cb099cf1bfe76b5e6edd718e5a/tools/tts_streaming.py) · [本机服务返回方式](../provider/server.py)

客户端回归使用实际 Hermes 配置加载器、工具注册表和 OpenAI SDK，在 HTTP 传输层给出已知音频响应，核对请求只发到本地地址、台词与导演描述完整、编码正确，并检查配置误指其他模型时会终止。复跑命令如下。[客户端测试](../integrations/hermes/test_client.py)

```powershell
& $hermesPython -B -X utf8 -m unittest discover -s integrations/hermes -p test_client.py -v
```
