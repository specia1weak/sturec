param(
    [string]$DatasetIndices = "2,0",
    [string]$BackboneIndices = "0,1,2,3,4,5,6,7",
    [string]$CustomSuffix = "msr",
    [string]$PythonBin = "python"
)

$ErrorActionPreference = "Continue"

$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$RootDir = Resolve-Path (Join-Path $ScriptDir "..\..")
$EntryScript = Join-Path $ScriptDir "msr-sumup.py"
$SrcPath = Join-Path $RootDir "src"

function Split-IndexList {
    param([string]$Value)
    return $Value.Split(",") |
        ForEach-Object { $_.Trim() } |
        Where-Object { $_ -ne "" }
}

$Datasets = Split-IndexList $DatasetIndices
$Backbones = Split-IndexList $BackboneIndices

Write-Host "[run-msr-grid] root=$RootDir"
Write-Host "[run-msr-grid] datasets=$DatasetIndices"
Write-Host "[run-msr-grid] backbones=$BackboneIndices"
Write-Host "[run-msr-grid] customsuffix=$CustomSuffix"
Write-Host "[run-msr-grid] python=$PythonBin"

$OldLocation = Get-Location
$OldPythonPath = $env:PYTHONPATH

try {
    Set-Location $RootDir
    if ([string]::IsNullOrWhiteSpace($OldPythonPath)) {
        $env:PYTHONPATH = $SrcPath
    } else {
        $env:PYTHONPATH = "$SrcPath;$OldPythonPath"
    }

    foreach ($DatasetIndex in $Datasets) {
        foreach ($BackboneIndex in $Backbones) {
            Write-Host "================================================================================"
            Write-Host "[run-msr-grid] dataset_index=$DatasetIndex backbone_index=$BackboneIndex"
            Write-Host "================================================================================"

            & $PythonBin $EntryScript `
                --datasetindex $DatasetIndex `
                --backboneindex $BackboneIndex `
                --customsuffix $CustomSuffix

            if ($LASTEXITCODE -ne 0) {
                Write-Warning "[run-msr-grid][skip] dataset_index=$DatasetIndex backbone_index=$BackboneIndex exit=$LASTEXITCODE"
                continue
            }
        }
    }
}
finally {
    Set-Location $OldLocation
    $env:PYTHONPATH = $OldPythonPath
}
