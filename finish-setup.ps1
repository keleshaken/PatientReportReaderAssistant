$ErrorActionPreference = 'Stop'
Set-Location $PSScriptRoot
$pythonPath = Join-Path $PSScriptRoot '.venv\Scripts\python.exe'
$ollamaPath = Join-Path $env:LOCALAPPDATA 'Programs\Ollama\ollama.exe'
Start-Transcript -Path (Join-Path $PSScriptRoot 'setup-progress.log') -Append
try {
    Write-Output 'Waiting for the existing package installation to finish...'
    $installProcesses = Get-CimInstance Win32_Process | Where-Object {
        $_.Name -match '^python.*\.exe$' -and
        $_.CommandLine -match 'pip install -r requirements.txt'
    }
    foreach ($installProcess in $installProcesses) {
        Wait-Process -Id $installProcess.ProcessId -ErrorAction SilentlyContinue
    }
    & $pythonPath -m pip install -r requirements.txt --timeout 20 --retries 3 --progress-bar off
    if ($LASTEXITCODE -ne 0) { throw 'Package installation failed. See setup-progress.log.' }
    Write-Output 'Waiting for the Ollama installation to finish...'
    while (-not (Test-Path $ollamaPath)) { Start-Sleep -Seconds 15 }
    $ollamaReady = $false
    while (-not $ollamaReady) {
        try {
            Invoke-RestMethod 'http://127.0.0.1:11434/api/version' | Out-Null
            $ollamaReady = $true
        } catch {
            if (-not (Get-Process ollama -ErrorAction SilentlyContinue)) {
                Start-Process -FilePath $ollamaPath -ArgumentList 'serve' -WindowStyle Hidden
            }
            Start-Sleep -Seconds 15
        }
    }
    & $ollamaPath pull qwen3:1.7b
    if ($LASTEXITCODE -ne 0) { throw 'Ollama model download failed. See setup-progress.log.' }
    Write-Output 'Starting the app at http://localhost:8501'
    & $pythonPath -m streamlit run app.py --server.address 127.0.0.1 --server.headless true --browser.gatherUsageStats false
    if ($LASTEXITCODE -ne 0) { throw 'Streamlit exited with an error.' }
} finally {
    Stop-Transcript
}
