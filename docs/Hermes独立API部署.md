# 场景一 · 独立部署 API，给 Hermes 长期提供 TTS

在一台有 NVIDIA GPU 的 Windows 或 Ubuntu x86-64 服务器上运行 Drama 3。Hermes 所在机器只需访问 HTTP。下面以 Windows 服务器 `192.168.1.241`、Ubuntu 客户端 `192.168.100.215` 为例。换机器时改成自己的服务器 IP 或域名。

- **对外地址**为 `http://192.168.1.241:8195/v1`。网关默认监听 `0.0.0.0:8195`，访问范围由防火墙或网络层管理。8196 是网关与同机原生引擎的内部端口。
- **从零部署**按本页选择一种操作系统、准备一个后端，先用内置音色生成文件，再装开机服务。整条路线独立于 ComfyUI。
- **12 GiB 显卡先用 `5lang`**。五语默认 12 GiB 档，专家权重约占 8.63 GiB 固定系统内存，还要为系统和其他程序留出内存。16 GiB 卡可设 `JR_DRAMA3_VRAM_GIB=16`。12/16 GiB 的历史数据来自 5090 限额实测，其他型号的速度以实机为准。[五语测试结果](五语推理.md#5090-开发机实测)

## 1. 在 Windows 从零安装

示例安装在 `F:\AI\JR-ComfyUI-Drama3`。安装 64 位 Git、可供服务账户运行的 64 位 Python 3.12 和 NVIDIA 驱动。在 PowerShell 执行 `nvidia-smi`、`git --version`、`py -3.12 --version`。从源码准备五语或 NVFP4 引擎时，另装 VS 2022 Build Tools 的「使用 C++ 的桌面开发」、Windows SDK、CMake、Ninja 和 CUDA Toolkit 13，确认 `nvcc --version` 可用。Q4 使用发布的 Vulkan 程序。已有完整 Windows 推理包时可复用包中对应的引擎、DLL 和权重。[Python](https://www.python.org/downloads/windows/) · [Git](https://git-scm.com/downloads/win) · [Visual Studio Build Tools](https://visualstudio.microsoft.com/visual-cpp-build-tools/) · [CUDA Toolkit 归档](https://developer.nvidia.com/cuda-toolkit-archive)

```powershell
New-Item -ItemType Directory -Force 'F:\AI' | Out-Null
Set-Location 'F:\AI'
git clone https://github.com/Goldlionren/JR-ComfyUI-Drama3.git
Set-Location '.\JR-ComfyUI-Drama3'
py -3.12 -m venv .venv
$py = (Resolve-Path '.\.venv\Scripts\python.exe').Path
$env:Path = "$(Split-Path $py);$env:Path"
& $py -m pip install -r .\tools\requirements-realtime.txt
```

**按所选后端执行一行准备命令。** 如果从零生成五语 GGUF，先在上述环境运行 `& $py -m pip install torch 'gguf==0.19.0' safetensors huggingface_hub`，并为源权重、转换临时文件和最终 GGUF 预留磁盘空间。已有五语 GGUF 时，将它和同名的 `.gguf.sha256` 放进项目的 `.runtime`，脚本会复用两者。已有完整 Windows 五语推理包时，保留包内目录结构、安装 Microsoft Visual C++ x64 运行库，确认驱动兼容 CUDA 13、CPU 支持 AVX2，即可跳到手动启动。[模型位置表](ComfyUI纯推理.md#model-paths) · [Windows 五语推理包](五语推理.md#windows-推理机)

| 后端 | Windows 准备命令 | 启动参数 |
|---|---|---|
| 五语（普通话、英语、日语、韩语、粤语） | `.\tools\prepare_realtime.ps1 -Backend 5lang` | `--backend 5lang` |
| NVFP4 | `.\tools\prepare_realtime.ps1 -Backend nvfp4` | `--backend nvfp4` |
| Q4_K | `.\tools\prepare_realtime.ps1 -Backend q4` | `--backend q4` |

先手动启动。下面以五语为例，另外两个后端将 `--backend 5lang` 换成表中的对应参数。首次冷启动包含权重校验、加载和预热，等到 `realtime ready` 再发请求。示例同时把音色库位置固定到开机服务使用的数据目录。

```powershell
$env:JR_DRAMA3_VOICE_DB = "$env:ProgramData\Drama3-Data\voices.sqlite3"
& $py -B .\tools\realtime_service.py start --backend 5lang
& $py -B .\tools\realtime_service.py status
(Invoke-RestMethod 'http://192.168.1.241:8195/health').data
```

## 2. 在 Ubuntu 从零安装

以 Ubuntu 24.04、Python 3.12 为例，以下命令在将来运行服务的普通用户下执行。安装服务器对应的 NVIDIA 驱动，确认 `nvidia-smi`。编译五语和 NVFP4 引擎请装 CUDA Toolkit 13 和兼容的宿主编译器，确认 `nvcc --version`。需要时将 `/usr/local/cuda/bin` 加入 PATH。Q4 使用驱动提供的 Vulkan 支持。若采用 Ubuntu 22.04，先准备 Python 3.11+ 和匹配的构建工具链。[NVIDIA Linux 安装指南](https://docs.nvidia.com/cuda/cuda-installation-guide-linux/) · [Ubuntu 22.04 的 Python 选择](安装部署.md)

```bash
sudo apt update
sudo apt install git curl unzip util-linux python3-venv build-essential cmake ninja-build libvulkan1 ffmpeg file
sudo install -d -o "$(id -un)" -g "$(id -gn)" /opt/JR-ComfyUI-Drama3 /var/lib/drama3
git clone https://github.com/Goldlionren/JR-ComfyUI-Drama3.git /opt/JR-ComfyUI-Drama3
cd /opt/JR-ComfyUI-Drama3
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r tools/requirements-realtime.txt
```

以下为五语全新安装。已有预转换权重和校验文件时可以跳过转换依赖那一行，仍需准备本系统的原生引擎。使用 NVFP4 或 Q4 时跳过转换依赖，并把两个命令里的 `5lang` 同时改为 `nvfp4` 或 `q4`。

```bash
python -m pip install torch 'gguf==0.19.0' safetensors huggingface_hub
bash tools/prepare_realtime.sh 5lang
export JR_DRAMA3_VOICE_DB=/var/lib/drama3/voices.sqlite3
python -B tools/realtime_service.py start --backend 5lang
python -B tools/realtime_service.py status
```

模型下载、构建与转换发生在安装阶段。开机服务只加载服务器上已经准备好的文件。Windows 与 Ubuntu 可以复用同一份 GGUF，原生引擎须与操作系统、GPU 和驱动相匹配。

## 3. 从 Hermes 机器验收

在客户端执行。Ubuntu API 服务器也用这组命令验证，只需替换服务器地址。`/health` 中应有 `active: true` 和所选后端。第二个请求会真正调用模型和 MP3 编码器。

```bash
API=http://192.168.1.241:8195
curl --noproxy '*' -fsS "$API/health"
curl --noproxy '*' -fsS "$API/v1/audio/speech" \
  -H 'Content-Type: application/json' \
  -d '{"model":"drama-3","voice":"zh","input":"语音服务已就绪。","response_format":"mp3"}' \
  -o /tmp/drama3-check.mp3 && file /tmp/drama3-check.mp3
```

在运行 Hermes 的机器执行 `hermes config path`，编辑现有 `tts` 段。先用内置 `zh` 验收，随后将 `voice` 换成目标服务器列出的完整音色 ID。`drama-3` 使用当前驻留后端。如需明确选择五语，填 `drama-3-5lang`。保存后重开 Hermes 会话。使用常驻 Hermes Gateway 时执行 `hermes gateway restart`。

```yaml
tts:
  provider: openai
  use_gateway: false
  openai:
    base_url: http://192.168.1.241:8195/v1
    api_key: not-needed
    model: drama-3
    voice: zh
    speed: 1.0
```

<a id="voice-database"></a>

## 4. 音色数据库的默认路径、改址和迁移

| 运行方式 | 8195 读取的库 |
|---|---|
| 独立 Git 克隆，未指定路径且未识别到 ComfyUI | `<仓库>/.runtime/provider-service/voices.sqlite3` |
| 识别到秋叶目录或仓库直接装在 `custom_nodes` | `<ComfyUI>/user/jr_drama3/voices.sqlite3` |
| 显式设置 `JR_DRAMA3_VOICE_DB` | 该变量指向的本机 SQLite **文件**。这是独立进程、目录链接和服务账户共用旧库时最可靠的配置 |
| 本页开机服务示例 | Windows 为 `%ProgramData%\Drama3-Data\voices.sqlite3`，Ubuntu 为 `/var/lib/drama3/voices.sqlite3` |

新安装可以直接使用内置音色。第一次通过 API 注册音色时才会创建数据库。旧的 `voice_id` 依赖原库中对应的参考录音和逐字稿。重装 Python、复制 GGUF 或 `git pull` 都不会迁移这些记录。

1. **认准旧库。** ComfyUI Desktop 的常见实例路径是 `F:\Comfy-Desktop\ComfyUI-Installs\default\ComfyUI\user\jr_drama3\voices.sqlite3`。画布使用自定义 `--user-directory` 时，从该目录下的 `jr_drama3` 找。在项目根目录用网关的 Python 执行下列命令，可显示该终端环境下选中的文件。
2. **迁移并授权。** 停止可能写入音色的网关和 ComfyUI，再复制旧库到目标位置，或使用 SQLite 在线备份。目标已存在另一份库时先备份并确认要保留哪份。服务账户需要能读取、写入数据库及其父目录。
3. **固定到启动环境。** 手动启动用 PowerShell 的 `$env:JR_DRAMA3_VOICE_DB = '完整路径'` 或 Bash 的 `export JR_DRAMA3_VOICE_DB=/完整路径/voices.sqlite3`。开机服务分别在下面的 WinSW XML 或 systemd 环境文件中设置。改完重启，再从客户端查音色列表。

```text
python -B -c "from provider.server import default_db; p=default_db(); print(p); print('exists =', p.is_file())"
```

```bash
curl --noproxy '*' -fsS http://192.168.1.241:8195/v1/audio/voices
```

若 MP3 用 `zh` 能生成，自定义 ID 却返回 `The local voice library is empty`，优先检查服务账户指向的数据库文件是否存在。若返回 `No local voice with ID`，则当前库里缺少该记录。可通过 `POST /v1/audio/voices` 注册 PCM16 WAV 与逐字稿。记录版本 ID 后，再填进 Hermes。[注册请求示例](中英实时语音.md#更换或注册人声) · [声音资产与备份](声音资产.md)

<a id="ubuntu-service"></a>

## 5. Ubuntu 开机自启 · systemd

以下示例采用系统级模板，实例名是刚才安装并成功试跑模型的 Linux 用户。工作目录、Python 和项目路径都固定在 `/opt/JR-ComfyUI-Drama3`。先停止手动实例，再交给一个服务管理器接管。

```bash
cd /opt/JR-ComfyUI-Drama3
.venv/bin/python -B tools/realtime_service.py stop
sudo install -m 0644 docs/services/drama3-tts@.service /etc/systemd/system/
sudoedit /etc/default/drama3-tts
```

在环境文件中保存下面三行。模型在独立磁盘时，也把所用的 `JR_DRAMA3_5LANG_MODEL` 等绝对路径写在这里。若 CUDA 动态库使用自定义搜索目录，在同一文件中明确填写所需的 `PATH` 或 `LD_LIBRARY_PATH`，服务不加载交互 shell 配置。

```ini
DRAMA3_BACKEND=5lang
JR_DRAMA3_VRAM_GIB=12
JR_DRAMA3_VOICE_DB=/var/lib/drama3/voices.sqlite3
```

```bash
UNIT="drama3-tts@$(id -un).service"
sudo systemctl daemon-reload
sudo systemctl enable --now "$UNIT"
systemctl is-enabled "$UNIT"
systemctl status "$UNIT" --no-pager
journalctl -u "$UNIT" -b -n 50 --no-pager
```

启动器返回时两个后台进程已就绪，因此 unit 使用 `Type=oneshot` 和 `RemainAfterExit=yes`。`active (exited)` 是预期状态。当前引擎健康状况以 `/health` 为准。此模板提供开机启动和受控启停，子进程异常退出后用 `restart` 恢复。以下管理命令按需单独执行。[项目 unit](services/drama3-tts@.service) · [systemd 服务类型](https://www.freedesktop.org/software/systemd/man/latest/systemd.service.html)

| 操作 | Ubuntu 命令（同一个终端的 `$UNIT`） |
|---|---|
| 停止 / 启动 | `sudo systemctl stop "$UNIT"` / `sudo systemctl start "$UNIT"` |
| 修改环境文件后重启 | `sudo systemctl restart "$UNIT"` |
| 取消开机启动并停止 | `sudo systemctl disable --now "$UNIT"` |

<a id="windows-service"></a>

## 6. Windows 开机自启 · Windows 服务

用 WinSW 2.12.0 将五语 API 注册为 `Drama3TTS`，无需用户登录。普通 `start` 预热完成后会退出，配套的 PowerShell 脚本会等待网关进程，Windows 服务管理器由此跟踪其生命周期。停止服务时 WinSW 调用项目启动器的 `stop`，释放网关和 GPU 引擎。[服务配置](services/Drama3TTS.xml) · [配套脚本](services/wait-gateway.ps1) · [WinSW 发行版](https://github.com/winsw/winsw/releases/tag/v2.12.0) · [WinSW 配置参考](https://github.com/winsw/winsw/blob/v2.12.0/doc/xmlConfigFile.md)

打开**管理员 PowerShell**，从已经手动试跑成功的独立安装继续。示例用 Windows 内置的 LocalService 账户。Python 基础安装、项目和模型必须能被该账户读取，尤其要注意安装在个人用户目录下的 Python。服务账户还要能使用 NVIDIA 驱动。

```powershell
$repo = 'F:\AI\JR-ComfyUI-Drama3'
Set-Location $repo
& "$repo\.venv\Scripts\python.exe" -B .\tools\realtime_service.py stop

$svc = "$env:ProgramData\Drama3-Service"
$data = "$env:ProgramData\Drama3-Data"
$state = "$repo\.runtime\provider-service"
New-Item -ItemType Directory -Force $svc,$state,"$data\logs" | Out-Null
Invoke-WebRequest -UseBasicParsing 'https://github.com/winsw/winsw/releases/download/v2.12.0/WinSW-x64.exe' -OutFile "$svc\Drama3TTS.exe"
Copy-Item .\docs\services\Drama3TTS.xml "$svc\Drama3TTS.xml"
icacls.exe $svc /grant '*S-1-5-19:(OI)(CI)RX'
icacls.exe $repo /grant '*S-1-5-19:(OI)(CI)RX'
icacls.exe $state /grant '*S-1-5-19:(OI)(CI)M'
icacls.exe $data /grant '*S-1-5-19:(OI)(CI)M'
notepad "$svc\Drama3TTS.xml"
```

确认 XML 的 `DRAMA3_ROOT`、`DRAMA3_BACKEND`、`JR_DRAMA3_VOICE_DB` 对应这次部署，再保存关闭。若模型在其他磁盘，可加一行 `<env name="JR_DRAMA3_5LANG_MODEL" value="D:\Models\Drama3\Drama3-Tier2-5Lang.gguf"/>`。服务账户需要能读取该 GGUF 及其校验文件。完整 Windows 推理包随引擎附带 CUDA DLL，自定义 CUDA 安装则需确保服务的 PATH 能找到相同 DLL。更新时保留部署目录中编辑过的 XML。

**已有 Desktop 音色库**时，可以继续使用原文件。在 XML 中将音色库那一行改成下面的值，并用管理员 PowerShell 给 LocalService 授予该库目录的修改权限。换了 Desktop 实例时相应替换两处路径。

```xml
<env name="JR_DRAMA3_VOICE_DB" value="F:\Comfy-Desktop\ComfyUI-Installs\default\ComfyUI\user\jr_drama3\voices.sqlite3"/>
```

```powershell
icacls.exe 'F:\Comfy-Desktop\ComfyUI-Installs\default\ComfyUI\user\jr_drama3' /grant '*S-1-5-19:(OI)(CI)M'
```

在同一个管理员窗口注册并启动服务。

```powershell
& "$svc\Drama3TTS.exe" install
Start-Service Drama3TTS
Get-Service Drama3TTS
(Invoke-RestMethod 'http://192.168.1.241:8195/health').data
```

首次启动可能仍在加载模型，稍后重试健康请求。常用命令如下，按需单独执行。

| 操作 | 管理员 PowerShell |
|---|---|
| 停止 / 启动 | `Stop-Service Drama3TTS` / `Start-Service Drama3TTS` |
| 修改 XML 中的音色库或模型后重启 | `Restart-Service Drama3TTS` |
| 取消开机启动并保留服务 | `Stop-Service Drama3TTS`，然后 `Set-Service Drama3TTS -StartupType Manual` |
| 恢复开机启动 | `Set-Service Drama3TTS -StartupType Automatic` |
| 卸载服务注册 | 先 `Stop-Service Drama3TTS`，再运行 `& "$svc\Drama3TTS.exe" uninstall` |

## 7. 更新、查看日志和验证重启

用管理器停止服务，进入原 Git 仓库执行 `git pull --ff-only origin main`，用该服务的 `.venv` 执行 `python -m pip install -r tools/requirements-realtime.txt`。发行说明涉及引擎或模型时，以原来的模型路径和同一后端重新运行准备脚本。Ubuntu 更新了 unit 模板时重新复制模板并执行 `systemctl daemon-reload`。Windows 的服务 XML 若有模板变化，合并到已安装的那一份。使用管理器重新启动。

- 网关和原生引擎日志分别是 `<仓库>/.runtime/provider-service/realtime.log` 和 `realtime-engine.log`。Windows 服务外层日志另在 `%ProgramData%\Drama3-Data\logs`。
- 冷启动验收要从 Hermes 所在机器发起。查询 `/health`，确认 `/v1/audio/voices` 中有自己的 ID，再用该 ID 生成一次 MP3。下次主机重启后重复这三项，就同时验证了开机服务、音色库和编码器。

| Hermes 机器上的结果 | 下一步查哪里 |
|---|---|
| 连接超时或被拒绝 | 在服务器查看监听状态。Windows 用 `Get-NetTCPConnection -State Listen -LocalPort 8195`，Ubuntu 用 `ss -ltn 'sport = :8195'`。网关正常时会看到 `0.0.0.0:8195`。随后检查两台机器之间的路由和防火墙。 |
| 返回 403 和 `Request outside the configured network` | HTTP 已到达一个应用。核对 Hermes 的地址、占用 8195 的进程和它使用的仓库版本。当前启动入口监听所有 IPv4 网卡，已取消读取旧版 `realtime-lan.txt`。更新后要重启实际提供服务的网关。[网关启动入口](../provider/realtime.py) |
| 请求 MP3 时返回 400 和 `Choose pcm (audio or sse) or wav (audio).` | 核对正在运行的网关版本。当前版本接受普通音频模式的 `response_format: mp3`。如果返回 503 并提示编码器不可用，在服务所用的 venv 中安装 `tools/requirements-realtime.txt`，重启后再执行第 3 节的 MP3 请求。 |
| 内置音色能朗读，换回自定义 ID 就报空库 | 按第 4 节核对服务配置中的 SQLite 路径。服务启动后从 Hermes 机器查 `/v1/audio/voices`，选 `source=library` 的完整 ID。 |
