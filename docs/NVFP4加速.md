# 用项目的 NVFP4 GGUF 加速语音生成

**本页适用于只做推理。** 在 Windows 或 Ubuntu 上使用本项目已转换的 **NVFP4 256E**，直接下载或复用 GGUF，再编译带配套补丁的 CUDA 推理引擎，以 `--backend nvfp4` 启动即可。可以跳过 FP8 十五分片、模型转换和训练数据准备。

服务在 GPU 上常驻模型，8195 接收请求并把生成的 PCM 分块送回客户端。ComfyUI 对应「中英实时语音」节点，Hermes 对应 `/v1/audio/speech`。如果目标是训练新权重，请看[训练准备与权重兼容性](安装部署.md#需要训练)。

## 1. 把 GGUF 放在哪台机器

只在**运行推理引擎的 GPU 服务器**上放一份模型。Windows 与 Ubuntu 使用相同的 22,244,242,848 字节文件。默认位置以 **`tools/realtime_service.py` 所属项目**为基准，和启动终端的当前目录无关。

| 服务器 | 默认位置举例 | 想复用独立模型盘时，设置 `JR_DRAMA3_NVFP4_MODEL` 为 |
|---|---|---|
| Windows，仓库在 `F:\AI\JR-ComfyUI-Drama3` | `F:\AI\JR-ComfyUI-Drama3\.runtime\Drama3-V2-78e5ab7-NVFP4-256E.gguf` | `D:\AI Models\Drama3\Drama3-V2-78e5ab7-NVFP4-256E.gguf` |
| Ubuntu，仓库在 `/home/alex/JR-ComfyUI-Drama3` | `/home/alex/JR-ComfyUI-Drama3/.runtime/Drama3-V2-78e5ab7-NVFP4-256E.gguf` | `/mnt/models/Drama3/Drama3-V2-78e5ab7-NVFP4-256E.gguf` |

如果直接克隆在 ComfyUI 插件目录，默认位置就是 `<ComfyUI>/custom_nodes/JR-ComfyUI-Drama3/.runtime/Drama3-V2-78e5ab7-NVFP4-256E.gguf`。使用联接或符号链接时，`.runtime` 位于被链接的项目中。自定义变量填**文件的绝对路径，含文件名**。Ubuntu 路径区分大小写；WSL 中访问 Windows D 盘要写 `/mnt/d/...`。给服务账号读取权限，使用准备脚本下载时还要能写入父目录。

模型来自固定快照 `06cfd87d3aa87cdd37e816fac6da485b8f6cbdb4`，预期 SHA-256 为 `5577beb99be7bc0f4c8324daea90e2f3c8765d17842db41016560fb4820312b1`。准备脚本可以从头下载，也可以复用手工下载的同一文件；其余引擎文件仍在项目自己的 `.runtime`。当前启动器按该指纹核验权重。[GGUF 和适配补丁的下载页](https://huggingface.co/WonderChord/Drama3-V2-NVFP4-256E-GGUF/tree/06cfd87d3aa87cdd37e816fac6da485b8f6cbdb4)

## 2. 准备并启动

先按[安装部署](安装部署.md)准备 Python、Git、完整 CUDA 13 Toolkit、当前平台的 C++ 工具链及 CMake/Ninja。精简安装还应包含 CUDA Runtime、cuBLAS 和 cuFFT 的开发库；配置阶段出现 `CUDA::cufft` 表示编译器已找到，但当前 Toolkit 缺少 cuFFT 开发库或 CMake 指向了另一套 Toolkit。标准编译目标为 `120a`。以下命令都在项目根目录执行。**使用默认位置时，跳过设置模型变量的那一行。** 已有模型放在自定义模型盘时只需设置变量，脚本校验通过后会复用它。

**Windows PowerShell**

```powershell
$py = (Resolve-Path '.venv\Scripts\python.exe').Path
$env:JR_DRAMA3_NVFP4_MODEL = 'D:\AI Models\Drama3\Drama3-V2-78e5ab7-NVFP4-256E.gguf'
& $py -m pip install -r tools/requirements-realtime.txt
.\tools\prepare_realtime.ps1 -Backend nvfp4
& $py -B tools/realtime_service.py start --backend nvfp4
(Invoke-RestMethod 'http://127.0.0.1:8195/health').data | Select-Object id,backend
(Get-Content -Raw -Encoding UTF8 '.runtime/provider-service/realtime-nvfp4.json' | ConvertFrom-Json).models[0].path
```

**Ubuntu Bash**

```bash
PY="$PWD/.venv/bin/python"
export JR_DRAMA3_NVFP4_MODEL="/mnt/models/Drama3/Drama3-V2-78e5ab7-NVFP4-256E.gguf"
"$PY" -m pip install -r tools/requirements-realtime.txt
bash tools/prepare_realtime.sh nvfp4
"$PY" -B tools/realtime_service.py start --backend nvfp4
curl -fsS http://127.0.0.1:8195/health
"$PY" -c "import json; print(json.load(open('.runtime/provider-service/realtime-nvfp4.json', encoding='utf-8'))['models'][0]['path'])"
```

健康响应应同时出现 `drama-3` 和 `nvfp4`。最后一行显示引擎配置实际使用的 GGUF 路径。首次启动会校验文件、加载并预热中文和英文。以后若更换模型路径或后端，先执行 `tools/realtime_service.py stop`，再在设置了新变量的终端启动。临时设置只对当前终端及子进程生效，作为系统服务运行时把变量写进该服务的配置。

## 3. 用自己的台词验证加速

Windows 和 Ubuntu 22.04（WSL2）在同一张 RTX 5090 上分别启动本系统的引擎，通过 8195 合成了「你好。See you soon.」。两次都使用本项目的 GGUF、内置 `zh` 参考与种子 184，均返回 44.1 kHz 单声道 PCM16 WAV。Ubuntu 的模型位于带空格的独立目录，已核对引擎配置指向该文件。首次安装可用同一句试听。[Windows 样音](samples/nvfp4-mixed.wav) · [Ubuntu 样音](samples/nvfp4-ubuntu-mixed.wav)

- **在画布用**：保持 8195 运行，在同一台主机的 ComfyUI 打开 [07 中英实时语音](../example_workflows/07-中英实时语音.json)，先选 `zh` 或 `en`，填台词并点击「启用本机流式试听」，再排队。节点输出最终 AUDIO 和延迟报告。克隆自己的参考音或对某句话加耳语时打开 [08 实时音色与语气](../example_workflows/08-实时音色与语气.json)。
- **从应用调用**：Base URL 填 `http://127.0.0.1:8195/v1`。其他机器用服务端实际配置的局域网 IP。请求 `response_format="pcm"` 可边收边播，`"wav"` 得到完整文件。两个平台的第一条合成命令见[安装部署](安装部署.md)，注册声音和 Hermes 配置见[实时 API](中英实时语音.md)。
- **测自己机器的速度**：用 `python tools/benchmark_realtime.py --output outputs/my-nvfp4-run` 请求实际的 8195 服务并保存样音和 `timings.json`。固定参考、台词和种子后比较首音延迟与 RTF（生成耗时 ÷ 音频时长，小于 1 表示快于播放）。历史 Windows RTX 5090 测得 65.48 秒混读在 12.87 秒生成，首段 PCM 约 107 毫秒。这是一组热运行结果，冷启动与其他 GPU 请在目标机测量。

GGUF 走 8195 量化引擎。画布「本地 FP8 教师」的模型目录、`JR_DRAMA3_MODEL_DIR`、FLA 开关以及 8193 教师网关都属于十五分片路线。需要上下文补录、多候选或教师的批量工作流时按[完整教师部署](安装部署.md)操作。两条路线共用一张显卡时，切换前可用「释放教师」或实时服务的 `stop` 腾出显存。

## 排查

| 现象 | 应核对的内容 |
|---|---|
| 下载器又在下载 22 GB | 检查准备和启动是否由同一环境运行；自定义变量是否指向已有文件，而不是父目录。在 PowerShell 用 `Test-Path -LiteralPath $env:JR_DRAMA3_NVFP4_MODEL`，在 Bash 用 `test -f "$JR_DRAMA3_NVFP4_MODEL"`。 |
| `SHA-256 mismatch` | 核对文件大小和快照。Windows 用 `Get-FileHash -Algorithm SHA256 -LiteralPath $env:JR_DRAMA3_NVFP4_MODEL`，Ubuntu 用 `sha256sum "$JR_DRAMA3_NVFP4_MODEL"`。默认安装时改填默认文件路径。 |
| GGUF 已就位，引擎仍无法启动 | 核对 `nvcc --version`、GPU、编译输出和 `.runtime/provider-service/realtime-engine.log`；引擎要使用固定源码和模型作者补丁。 |
| 健康响应显示 `q4`，或节点仍校验十五分片 | 在 8195 停止旧服务再启动 NVFP4；画布选示例 07/08 并重新排队。 |
