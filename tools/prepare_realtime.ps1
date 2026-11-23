[CmdletBinding()]
param([ValidateSet('q4', 'nvfp4')] [string]$Backend = 'nvfp4',
      [string]$VsInstall = '')
$ErrorActionPreference = 'Stop'
$runtime = Join-Path (Split-Path $PSScriptRoot -Parent) '.runtime'
New-Item -ItemType Directory -Force -Path $runtime | Out-Null

function Get-CheckedFile($url, $name, $sha, $destination = '') {
    $dest = if ($destination) { $destination } else { Join-Path $runtime $name }
    [IO.Directory]::CreateDirectory([IO.Path]::GetDirectoryName($dest)) | Out-Null
    if (-not (Test-Path -LiteralPath $dest)) {
        $partial = "$dest.partial"
        & curl.exe --fail --location --retry 3 --continue-at - --output $partial $url
        if ($LASTEXITCODE) { throw "Download failed: $name" }
        if ((Get-FileHash -LiteralPath $partial).Hash -ne $sha) { throw "SHA-256 mismatch: $partial" }
        Move-Item -LiteralPath $partial -Destination $dest
    } elseif ((Get-FileHash -LiteralPath $dest).Hash -ne $sha) {
        throw "SHA-256 mismatch: $dest"
    }
    return $dest
}

if ($Backend -eq 'q4') {
    $name = 'Drama-35B-A3B-v3-Q4_K.gguf'
    $repo = 'https://huggingface.co/sakasegawa/drama-3-GGUF/resolve/9a01354d04ab6eb521bc9cdfc56ab6824c474af2'
    $null = Get-CheckedFile "$repo/$name" $name '2a57b14c9b208a2ff833ab7305b920ca8f154ad807004a52d96a86a90a55523e'
    $zip = Get-CheckedFile 'https://github.com/nyosegawa/speech.cpp/releases/download/v0.11.2/speech-0.11.2-windows-x64-vulkan.zip' 'speechcpp-windows.zip' '2cf46f0351a2a90b9ce0bfb83cf63ac209ca31c88df9a5bc0126898a20a58586'
    Expand-Archive -LiteralPath $zip -DestinationPath (Join-Path $runtime 'speechcpp') -Force
    return
}

$name = 'Drama3-V2-78e5ab7-NVFP4-256E.gguf'
$repo = 'https://huggingface.co/WonderChord/Drama3-V2-NVFP4-256E-GGUF/resolve/06cfd87d3aa87cdd37e816fac6da485b8f6cbdb4'
$modelPath = $env:JR_DRAMA3_NVFP4_MODEL
if ($modelPath -and $modelPath -notmatch '^(?:[A-Za-z]:[\\/]|\\\\[^\\]+\\[^\\]+\\)') {
    throw 'JR_DRAMA3_NVFP4_MODEL must be an absolute path to the GGUF file.'
}
$null = Get-CheckedFile "$repo/$name" $name '5577beb99be7bc0f4c8324daea90e2f3c8765d17842db41016560fb4820312b1' $modelPath
$patch = Get-CheckedFile "$repo/drama3-nvfp4-audiocpp.patch" 'drama3-nvfp4-audiocpp.patch' 'bbb0742a2beb7c06a7592efaa5dca74f8bb1ba3f25da57ca9d882b874f082c09'
$src = Join-Path $runtime 'audio.cpp'
$commit = '066ea9862f41f4d7bcf996d5f00e464d8d28ace5'
if (-not (Test-Path -LiteralPath $src)) {
    & git -c core.longpaths=true clone --filter=blob:none --no-checkout https://github.com/0xShug0/audio.cpp.git $src
    if ($LASTEXITCODE) { throw 'Clone failed' }
    & git -C $src checkout --detach $commit
    if ($LASTEXITCODE) { throw 'Checkout failed' }
}
if ((& git -C $src rev-parse HEAD) -ne $commit) { throw "Expected $commit in $src" }
& git -C $src -c submodule.external/audio.cpp-server-frontends.url=https://github.com/0xShug0/audio.cpp-server-frontends.git submodule update --init --depth 1
if ($LASTEXITCODE) { throw 'Submodule fetch failed' }
& git -C $src apply --reverse --check --quiet $patch
if ($LASTEXITCODE) {
    & git -C $src apply $patch
    if ($LASTEXITCODE) { throw 'NVFP4 patch failed' }
}
$params = @{Preset='windows-cuda-release'; ModelSet='custom'; Models='fish_drama3'; Target='audiocpp_server'; Jobs=12; CudaArchitectures='120a'}
if ($VsInstall) { $params.VsInstall = $VsInstall }
& (Join-Path $src 'scripts/build_windows.ps1') @params
if ($LASTEXITCODE) { throw 'CUDA build failed' }
