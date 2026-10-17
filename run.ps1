param([ValidateSet('demo','reference','qwen')][string]$Mode='demo',[int]$Port=8000)
$ErrorActionPreference='Stop'
Set-Location -LiteralPath $PSScriptRoot
if (!(Test-Path -LiteralPath '.venv\Scripts\python.exe')) {
  py -3.11 -m venv .venv
  if ($LASTEXITCODE -ne 0) { throw 'Install Python 3.11, then run again.' }
  & .venv\Scripts\python.exe -m pip install -r requirements.txt
  if ($LASTEXITCODE -ne 0) { throw 'Dependency installation failed.' }
}
if ($Mode -eq 'demo' -and !(Test-Path -LiteralPath 'models\demo.npz')) {
  & .venv\Scripts\python.exe -m scripts.make_demo
  if ($LASTEXITCODE -ne 0) { throw 'Demo setup failed.' }
}
if (!$env:FORTE_API_KEY) {
  $env:FORTE_API_KEY = & .venv\Scripts\python.exe -c "import secrets; print(secrets.token_urlsafe(24))"
  Write-Host "Temporary local access key: $env:FORTE_API_KEY"
  Write-Host 'Copy it into the review page. Keep the jury key private.'
}
$env:FORTE_ENGINE=$Mode
if ($Mode -eq 'demo') { $env:FORTE_MODEL='models/demo.npz' }
if ($Mode -eq 'reference') { $env:FORTE_MODEL='models/reference.npz' }
if ($Mode -eq 'qwen') { $env:FORTE_MODEL='models/qwen3-vl-2b' }
& .venv\Scripts\python.exe -m uvicorn app.main:create_app --factory --host 127.0.0.1 --port $Port --workers 1
