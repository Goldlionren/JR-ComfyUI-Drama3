# 场景二 · 在 ComfyUI 用五语、NVFP4 或 Q4 纯推理

ComfyUI 提供节点和画布，同机的 **8195 网关**把请求交给加载 GGUF 的原生引擎。浏览器可以在其他电脑。重启 ComfyUI 和启停网关是两个独立操作。单句配音、参考录音克隆和自动朗读都可以直接使用现成权重。

## 1. 把插件装到正在运行的 ComfyUI

以下 Windows 示例对应 Desktop 的 `default` 实例。刚安装 ComfyUI 时先启动一次并确认画布能打开。如果还没有 ComfyUI，可从官方 Desktop 或手动安装开始。Ubuntu 源码安装示例假设含 `main.py` 的目录为 `~/ComfyUI`，使用它自己的 Python 环境。[ComfyUI 官方安装文档](https://docs.comfy.org/installation/manual_install)

**Windows Desktop。** 在 ComfyUI Desktop 的 Terminal 里执行。已有 `F:\AI\JR-ComfyUI-Drama3` 时跳过 clone。已有同名链接时检查 `Target`，避免把插件装两份。

```powershell
$comfy = 'F:\Comfy-Desktop\ComfyUI-Installs\default\ComfyUI'
$repo = 'F:\AI\JR-ComfyUI-Drama3'
$py = "$comfy\.venv\Scripts\python.exe"
if (-not (Test-Path $repo)) {
    New-Item -ItemType Directory -Force 'F:\AI' | Out-Null
    git clone https://github.com/Goldlionren/JR-ComfyUI-Drama3.git $repo
}
$link = "$comfy\custom_nodes\JR-ComfyUI-Drama3"
if (-not (Test-Path $link)) { New-Item -ItemType Junction -Path $link -Target $repo }
Get-Item $link | Select-Object FullName,Target
Set-Location $repo
& $py -m pip install -r .\tools\requirements-realtime.txt scipy soundfile safetensors
```

**Ubuntu。** 下面将仓库直接放在 `custom_nodes`。如果已经单独克隆，改用 `ln -s /实际仓库路径 "$COMFY/custom_nodes/JR-ComfyUI-Drama3"`，随后在该仓库操作。已有 ComfyUI 使用其他 venv 时替换 `PY`。

```bash
COMFY="$HOME/ComfyUI"
REPO="$COMFY/custom_nodes/JR-ComfyUI-Drama3"
PY="$COMFY/.venv/bin/python"
git clone https://github.com/Goldlionren/JR-ComfyUI-Drama3.git "$REPO"
cd "$REPO"
"$PY" -m pip install -r tools/requirements-realtime.txt scipy soundfile safetensors
```

安装后重启 ComfyUI 并刷新画布。搜索「Drama 3 · 实时语音」或「Drama 3 · 语音模型」。第一次验证可用 07 直接朗读，或用 01 的模型下拉框带动后续制作节点。已有工作流保留原节点 ID、连线和 FP8 默认值。

<a id="model-paths"></a>

## 2. 模型和引擎放在哪里

`<项目>` 是含 `tools/realtime_service.py` 的**真实仓库目录**。例如上面 Windows 安装的 `<项目>/.runtime` 就是 `F:\AI\JR-ComfyUI-Drama3\.runtime`。每行是一个可独立准备的后端。

| 画布 / API 的模型 ID | 启动参数 | 相对 `<项目>/.runtime/` 的权重文件 | 另存磁盘用的环境变量 |
|---|---|---|---|
| `drama-3-5lang` | `5lang` | `Drama3-Tier2-5Lang.gguf` 和 `Drama3-Tier2-5Lang.gguf.sha256`，权重约 14.11 GB | `JR_DRAMA3_5LANG_MODEL` |
| `drama-3-nvfp4` | `nvfp4` | `Drama3-V2-78e5ab7-NVFP4-256E.gguf`，约 22.24 GB | `JR_DRAMA3_NVFP4_MODEL` |
| `drama-3-q4` | `q4` | `Drama-35B-A3B-v3-Q4_K.gguf`，约 20.79 GB | `JR_DRAMA3_Q4_MODEL` |

- **引擎。** 五语和 NVFP4 共用 audio.cpp。Windows 为 `.runtime/audio.cpp/build/windows-cuda-release/bin/audiocpp_server.exe`，Ubuntu 为 `.runtime/audio.cpp/build/linux-cuda-release/bin/audiocpp_server`。Q4 使用 `.runtime/speechcpp/speech.exe` 或 `.runtime/speechcpp/speech`。Windows 和 Linux 可复制同一份 GGUF，需分别准备各自的可执行文件。
- **内置参考音色。** 普通话、英语在 `example_workflows/jr-benchmark-references/`，日语、韩语、粤语在 `resources/five-lang-voices/`。随 Git 一起分发。自定义音色在 SQLite 库里，见下文。
- **从源转换五语时**，原始检查点默认下载到 `.runtime/five-lang-source/`。转换后日常推理读取上表中的 GGUF 和校验文件。五语的 12 GiB 档会把部分权重放进系统内存，实测数据和 16 GiB 档的设置见[五语推理指南](五语推理.md)。
- **可选权重。** 离线转写默认使用 `.runtime/aux/whisper-small.pt`，也搜索 `<ComfyUI models>/whisper/small.pt` 和用户的 Whisper 缓存。选择 FP8 教师工作流时，官方十五分片默认在 `.runtime/teacher/`。翻译、分离、说话人分析和口型节点各自的下载、完整文件名和覆盖变量见[模型总表](模型.md)。三个 GGUF 后端可独立于这些可选模型工作。

要复用模型盘，将对应变量设为 **GGUF 文件的绝对路径**，准备脚本和运行网关时使用同一环境。五语校验文件放在 GGUF 旁边。下面以单独的模型盘为例。

```powershell
$env:JR_DRAMA3_5LANG_MODEL = 'D:\Models\Drama3\Drama3-Tier2-5Lang.gguf'
```

```bash
export JR_DRAMA3_5LANG_MODEL=/mnt/models/Drama3/Drama3-Tier2-5Lang.gguf
```

## 3. 准备一个后端，然后启动

已有匹配当前系统的完整推理包时，把 GGUF、校验文件、对应引擎及配套库按原目录放到真实仓库。保留本机的服务状态和音色库。已有 GGUF 但缺少引擎时仍要运行准备脚本。从全新 clone 准备五语，需要转换器的 Python 依赖。已有五语 GGUF 和校验文件时可跳过该项。CUDA/Vulkan、编译工具和系统依赖按[Windows 和 Ubuntu 准备步骤](Hermes独立API部署.md)安装。

| 想装的后端 | Windows（在项目根目录） | Ubuntu（在项目根目录） |
|---|---|---|
| 五语 | `.\tools\prepare_realtime.ps1 -Backend 5lang` | `bash tools/prepare_realtime.sh 5lang` |
| NVFP4 | `.\tools\prepare_realtime.ps1 -Backend nvfp4` | `bash tools/prepare_realtime.sh nvfp4` |
| Q4_K | `.\tools\prepare_realtime.ps1 -Backend q4` | `bash tools/prepare_realtime.sh q4` |

从源转换五语时，准备脚本会调用 PATH 中的 `python`。Windows Desktop 的 `(.venv)` 终端可运行 `python -c "import sys; print(sys.executable)"` 与 `$py` 核对。必要时先执行 `$env:Path = "$(Split-Path $py);$env:Path"`。Ubuntu 可执行 `source "$(dirname "$PY")/activate"` 激活上面选用的环境。已安装 PyTorch 的 ComfyUI 环境用该解释器执行 `python -m pip install 'gguf==0.19.0' safetensors huggingface_hub` 即可补齐转换依赖。

**让画布与 API 共用音色。** 在 ComfyUI 默认用户目录中，节点保存到 `user/jr_drama3/voices.sqlite3`。Desktop 通过 junction 安装时，单独启动的网关可能回退到插件的 `.runtime/provider-service/voices.sqlite3`。最稳妥的方式是把网关指向画布实际使用的库。Windows 示例同时写入当前 Terminal 和当前用户的持久环境。已有自定义库时请把下面路径换成它。使用 `--user-directory` 时从实际用户目录计算。

```powershell
Set-Location 'F:\AI\JR-ComfyUI-Drama3'
$comfy = 'F:\Comfy-Desktop\ComfyUI-Installs\default\ComfyUI'
$py = "$comfy\.venv\Scripts\python.exe"
$env:JR_DRAMA3_VOICE_DB = "$comfy\user\jr_drama3\voices.sqlite3"
[Environment]::SetEnvironmentVariable('JR_DRAMA3_VOICE_DB', $env:JR_DRAMA3_VOICE_DB, 'User')
& $py -B .\tools\realtime_service.py start --backend 5lang
& $py -B .\tools\realtime_service.py status
(Invoke-RestMethod 'http://192.168.1.241:8195/health').data | Select-Object id,backend,active
```

```bash
COMFY="$HOME/ComfyUI"
REPO="$COMFY/custom_nodes/JR-ComfyUI-Drama3"
PY="$COMFY/.venv/bin/python"
cd "$REPO"
export JR_DRAMA3_VOICE_DB="$COMFY/user/jr_drama3/voices.sqlite3"
"$PY" -B tools/realtime_service.py start --backend 5lang
"$PY" -B tools/realtime_service.py status
curl -fsS http://192.168.1.241:8195/health
```

上面两种平台均以 `192.168.1.241` 举例，请用 GPU 服务器实际地址。如果服务已由 systemd 或 Windows 服务接管，使用[服务管理器的启停命令](Hermes独立API部署.md)，并将相同音色库路径写进它的配置。

## 4. 在画布选模型、生成和换音色

1. 导入[07 · 中英实时语音](../example_workflows/07-中英实时语音.json)。在「Drama 3 · 实时语音」的 `model` 中选已准备的具体模型 ID，`voice` 先选 `zh`，输入「今天的语音测试开始了。」并运行队列。从节点预览试听。直播试听需先点「启用本机流式试听」。
2. 切换模型时选另一个已安装的 ID，重新执行节点即可。一次常驻一个引擎，切换和朗读期间并发请求可能返回 429。旧工作流中的 `drama-3` 会沿用当前引擎。也可在客户端查询 `GET /v1/models` 的 `installed`、`active` 字段，或向 `POST /v1/models/select` 提交 `{"model":"drama-3-q4"}`。
3. 用自己的声音时打开[08 · 实时音色与语气](../example_workflows/08-实时音色与语气.json)，连接参考录音，并填写录音中**实际说出的逐字稿**。新台词填 `text`。离线转写节点的 `model_path` 可留空，在仓库根目录执行 `python -B jr_analysis_prepare.py whisper` 准备转写权重，缺少 `whisper` 模块时在 ComfyUI 环境安装 `openai-whisper`。五语 GGUF 由实时语音节点选取。
4. 长期复用音色可执行「保存音色版本」，复制完整 `voice_id`。用 `GET /v1/audio/voices` 验证该 ID 出现在网关的声音列表，再给 Hermes 使用。[音色保存、查询与迁移](声音资产.md)

**候选、逐句和长文。** 在工作流里的「语音模型（FP8 / GGUF）」选择具体 GGUF，将原有 `teacher` 输出接到生成、候选、逐句、长文、回听重录、实时追加或固定基准节点。换模型会建立独立的候选缓存。逐句和长文的 `continuity=auto` 对 Q4 逐句复用角色参考，对五语、NVFP4 和 FP8 使用上一段历史。旧图如果已经保存 `linked`，切到 Q4 时手动改成 `auto` 或 `independent`。将模型输出接到「释放语音模型」即可卸载当前所选引擎，下次合成会重新加载。

**多角色与返修。** 用 GGUF 时把台本的 `text`、`cast` 接到「本地语音生成」，模式选 `line`，每个角色可绑定一份参考音。五语和 NVFP4 的 `continue` / `revise` 接同一份权重生成的上一段 `history`。原生整场 `scene_json` 和双侧上下文局部补录仍选 FP8。各节点适用范围见[兼容性审查](节点兼容性审查.md)。

## 5. 停止、升级与验收

**只停止量化服务**会释放它占用的显存。Windows 在项目根目录执行 `& $py -B .\tools\realtime_service.py stop`，Ubuntu 执行 `"$PY" -B tools/realtime_service.py stop`。停止 ComfyUI 用 Desktop 的 Stop 或源码终端的 Ctrl+C。两者单独管理。

升级插件前，等当前生成结束，停止 8195 和 ComfyUI。以下沿用前面设置的 `$py` 或 `$PY`。在真实插件仓库执行，确认 pull 成功后再继续。如果本地有编辑，先用 `git status` 查看并保留。

```powershell
Set-Location 'F:\AI\JR-ComfyUI-Drama3'
git pull --ff-only origin main
if ($LASTEXITCODE -eq 0) {
    & $py -m pip install -r .\tools\requirements-realtime.txt
    if ($LASTEXITCODE -eq 0) { & $py -B .\tools\realtime_service.py start --backend 5lang }
}
```

```bash
cd "$REPO"
git pull --ff-only origin main &&
  "$PY" -m pip install -r tools/requirements-realtime.txt &&
  "$PY" -B tools/realtime_service.py start --backend 5lang
```

若版本说明更新了原生引擎或模型，在 `start` 前先用原后端重新运行准备脚本。Git 管理节点、工作流和内置参考音，GGUF、`.runtime` 编译产物及个人音色库单独保存。最后启动 ComfyUI、刷新浏览器。确认 `/health` 显示所选后端、07 能播放内置音色、自己的 `voice_id` 能生成音频。
