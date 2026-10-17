param(
    [ValidateSet('demo','reference','qwen')][string]$Engine='demo',
    [string]$Model='',
    [string]$Runtime='',
    [ValidateRange(1024,65535)][int]$Port=8000,
    [string]$Python='.venv\Scripts\python.exe'
)
$ErrorActionPreference='Stop'
Set-Location -LiteralPath (Split-Path -Parent $PSScriptRoot)
if (!(Test-Path -LiteralPath $Python)) { throw 'Prepare the Python 3.11 environment first, or pass -Python with its path.' }
if (!$Runtime) { $Runtime="runtime/$Engine-service" }
if (!$Model) {
    $Model=@{demo='models/demo.npz'; reference='models/reference.npz'; qwen='models/qwen3-vl-2b'}[$Engine]
}
if ($Engine -eq 'demo' -and !(Test-Path -LiteralPath $Model)) {
    & $Python -m scripts.make_demo
    if ($LASTEXITCODE -ne 0) { throw 'Synthetic fixture setup failed.' }
}
if (!(Test-Path -LiteralPath $Model)) { throw 'Model files are missing. Finish model setup first.' }
$runtimePath=$ExecutionContext.SessionState.Path.GetUnresolvedProviderPathFromPSPath($Runtime)
[System.IO.Directory]::CreateDirectory($runtimePath) | Out-Null
$keyPath=Join-Path $runtimePath 'access-key.txt'
if (!(Test-Path -LiteralPath $keyPath)) {
    $secret=& $Python -c 'import secrets; print(secrets.token_urlsafe(24))'
    if ($LASTEXITCODE -ne 0) { throw 'Could not generate an access key.' }
    [System.IO.File]::WriteAllText($keyPath, $secret.Trim())
}
$env:FORTE_API_KEY=(Get-Content -LiteralPath $keyPath -Raw).Trim()
if ($env:FORTE_API_KEY.Length -lt 16) { throw 'Access key must contain at least 16 characters.' }
$env:FORTE_ENGINE=$Engine
$env:FORTE_MODEL=(Resolve-Path -LiteralPath $Model).Path
$env:FORTE_RUNTIME=$runtimePath
Write-Host "Service: http://127.0.0.1:$Port | readiness: /ready"
Write-Host "Private key file: $keyPath"
Write-Host 'Keep this terminal running. Startup includes a bounded model check.'
& $Python -m uvicorn app.main:create_app --factory --host 127.0.0.1 --port $Port --workers 1
exit $LASTEXITCODE
