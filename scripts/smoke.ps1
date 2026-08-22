<#
.SYNOPSIS
    Windows entry point for the fast suite. Mirrors `make smoke`.

.DESCRIPTION
    Prefers the repo's .venv interpreter so the ROCm torch build is the one under test.
    All logic lives in scripts/smoke.py; this only picks an interpreter.
#>
[CmdletBinding()]
param(
    [double] $Budget = 120,
    [switch] $VerboseOutput
)

$ErrorActionPreference = 'Stop'
$repo = Split-Path -Parent $PSScriptRoot
$venvPython = Join-Path $repo '.venv\Scripts\python.exe'
$python = if (Test-Path $venvPython) { $venvPython } else { 'python' }

$smokeArgs = @((Join-Path $repo 'scripts\smoke.py'), '--budget', $Budget)
if ($VerboseOutput) { $smokeArgs += '--verbose' }

& $python @smokeArgs
exit $LASTEXITCODE
