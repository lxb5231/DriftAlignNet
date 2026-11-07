$ErrorActionPreference = 'Stop'
$pythonCommand = if ($env:PYTHON) { $env:PYTHON } else { 'python' }
& $pythonCommand "$PSScriptRoot/run_dataset.py" energy @args
exit $LASTEXITCODE
