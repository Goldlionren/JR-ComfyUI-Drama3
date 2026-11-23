param(
    [string]$ComfyRoot = 'F:\ComfyUI-aki-v3\ComfyUI'
)
$ErrorActionPreference = 'Stop'
$source = (Resolve-Path (Join-Path $PSScriptRoot '..')).Path
$custom = (Resolve-Path (Join-Path $ComfyRoot 'custom_nodes')).Path
$destination = Join-Path $custom 'JR-ComfyUI-Drama3'
if (Get-Item -LiteralPath $destination -Force -ErrorAction SilentlyContinue) {
    $existing = Get-Item -LiteralPath $destination -Force
    if ($existing.LinkType -ne 'Junction' -or $existing.Target -ne $source) {
        throw "Existing installation at $destination belongs to a different checkout."
    }
    Write-Output "Already linked: $destination"
} else {
    New-Item -ItemType Junction -Path $destination -Target $source | Out-Null
    Write-Output "Installed: $destination -> $source"
}
