param(
    [string]$Source = '',
    [ValidateSet('nano', 'small', 'both')][string]$Variant = 'both',
    [ValidateSet('full_frame', 'overlap0', 'overlap128', 'overlap256', 'all')][string]$Mode = 'all',
    [string]$Device = 'cpu',
    [string]$Output = '',
    [string]$PythonExe = ''
)
$ErrorActionPreference = 'Stop'
$demoPython = Join-Path $PSScriptRoot '.venv-demo\Scripts\python.exe'
if (-not (Test-Path -LiteralPath $demoPython)) {
    if ($PythonExe) {
        & $PythonExe -c "import sys; assert sys.version_info[:2] in [(3,11),(3,12)], 'Python 3.11 or 3.12 required'"
        if ($LASTEXITCODE -ne 0) { throw 'Use a Python 3.11 or 3.12 executable.' }
        & $PythonExe -m venv (Join-Path $PSScriptRoot '.venv-demo')
    } else {
        py -3.12 -m venv (Join-Path $PSScriptRoot '.venv-demo')
    }
    if ($LASTEXITCODE -ne 0) { throw 'Install Python 3.12, or pass -PythonExe with the path to Python 3.11/3.12.' }
}
& $demoPython -m pip install torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cpu
if ($LASTEXITCODE -ne 0) { throw 'PyTorch installation failed.' }
& $demoPython -m pip install -e "${PSScriptRoot}[yolo]"
if ($LASTEXITCODE -ne 0) { throw 'Demo installation failed.' }
if (-not $Source) {
    & $demoPython (Join-Path $PSScriptRoot 'scripts\download_demo_sample.py')
    if ($LASTEXITCODE -ne 0) { throw 'Sample download failed; see docs/camera_ready.md for manual download.' }
    $Source = Join-Path $PSScriptRoot 'data\samples\transport.jpg'
}
if (-not $Output) { $Output = Join-Path $PSScriptRoot 'outputs\demo' }
& $demoPython (Join-Path $PSScriptRoot 'scripts\run_demo.py') --source $Source --variant $Variant --mode $Mode --device $Device --output $Output
if ($LASTEXITCODE -ne 0) { throw 'Demo failed.' }
