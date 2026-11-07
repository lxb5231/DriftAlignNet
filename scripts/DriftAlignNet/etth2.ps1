$ErrorActionPreference = 'Stop'
$pythonCommand = if ($env:PYTHON) { $env:PYTHON } else { 'python' }
& $pythonCommand "$PSScriptRoot/run_dataset.py" etth2 @args
exit $LASTEXITCODE
