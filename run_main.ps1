$ErrorActionPreference = 'Stop'
$pythonCommand = if ($env:PYTHON) { $env:PYTHON } else { 'python' }
& $pythonCommand "$PSScriptRoot/scripts/DriftAlignNet/run_dataset.py" @args
exit $LASTEXITCODE
