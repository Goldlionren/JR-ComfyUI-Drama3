# WinSW needs a foreground process for the lifetime of the HTTP gateway.
$ErrorActionPreference = 'Stop'
Set-Location -LiteralPath $env:DRAMA3_ROOT
& '.\.venv\Scripts\python.exe' -B '.\tools\realtime_service.py' start --backend $env:DRAMA3_BACKEND
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
$record = Get-Content -LiteralPath '.\.runtime\provider-service\realtime.json' -Raw | ConvertFrom-Json
$gateway = Get-Process -Id $record.pid -ErrorAction Stop
$gateway.WaitForExit()
