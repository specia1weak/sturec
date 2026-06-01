param(
    [int]$IntervalSeconds = 30,
    [int]$MaxIterations = 0
)

[Console]::OutputEncoding = [System.Text.UTF8Encoding]::new($false)
$OutputEncoding = [System.Text.UTF8Encoding]::new($false)

$workspaceRoot = Split-Path -Parent $PSScriptRoot
$boardRoot = $PSScriptRoot
$dirs = @("1_Todo", "2_Doing", "3_Done", "4_Block")
$logDir = Join-Path -Path $workspaceRoot -ChildPath "logs"
$logFile = Join-Path -Path $logDir -ChildPath "agent1_status.log"
$latestFile = Join-Path -Path $logDir -ChildPath "agent1_latest_status.txt"
$snapshotFile = Join-Path -Path $PSScriptRoot -ChildPath ".agent1_snapshot.json"

if (-not (Test-Path -LiteralPath $logDir)) { New-Item -ItemType Directory -Path $logDir -Force | Out-Null }

$initializeBaseline = -not (Test-Path -LiteralPath $snapshotFile)
$previous = @{}
if (Test-Path -LiteralPath $snapshotFile) {
    try { $previous = Get-Content -LiteralPath $snapshotFile -Raw -Encoding UTF8 | ConvertFrom-Json -AsHashtable } catch {
        $previous = @{}
        $initializeBaseline = $true
    }
}

function Get-DirSnapshot {
    $result = @{}
    foreach ($d in $dirs) {
        $path = Join-Path -Path $boardRoot -ChildPath $d
        $files = @()
        if (Test-Path -LiteralPath $path) {
            $files = Get-ChildItem -LiteralPath $path -File | Sort-Object Name | ForEach-Object {
                @{
                    Name = $_.Name
                    LastWrite = $_.LastWriteTime.ToString("yyyy-MM-dd HH:mm:ss")
                    Length = [int64]$_.Length
                }
            }
        }
        $result[$d] = $files
    }
    return $result
}

function Write-LogLine {
    param([string]$Line)
    $time = Get-Date -Format "yyyy-MM-dd HH:mm:ss"
    $entry = "[$time] $Line"
    Add-Content -LiteralPath $logFile -Value $entry -Encoding UTF8
    Write-Host $entry
}

function Get-DirMaps {
    param([hashtable]$Snapshot)

    $maps = @{}
    foreach ($d in $dirs) {
        $map = @{}
        $rows = @()
        if ($Snapshot.ContainsKey($d) -and $null -ne $Snapshot[$d]) {
            $rows = @($Snapshot[$d])
        }
        foreach ($row in $rows) {
            $map[$row["Name"]] = $row
        }
        $maps[$d] = $map
    }
    return $maps
}

function Write-LatestStatus {
    param(
        [string[]]$Summary,
        [string[]]$Changes
    )

    $lines = @()
    $lines += "LastCheck: $(Get-Date -Format "yyyy-MM-dd HH:mm:ss")"
    $lines += "Status: $($Summary -join ' | ')"
    if ($Changes.Count -gt 0) {
        $lines += "Changes:"
        foreach ($change in $Changes) {
            $lines += "  $change"
        }
    } else {
        $lines += "Changes: none"
    }

    Set-Content -LiteralPath $latestFile -Value $lines -Encoding UTF8
}

Write-LogLine "=== Agent1 Monitor STARTED (interval: ${IntervalSeconds}s, maxIterations: ${MaxIterations}) ==="

$iteration = 0

while ($true) {
    $current = Get-DirSnapshot
    $changes = New-Object System.Collections.Generic.List[string]
    $highlights = New-Object System.Collections.Generic.List[string]
    $previousMaps = Get-DirMaps -Snapshot $previous
    $currentMaps = Get-DirMaps -Snapshot $current
    $addedByName = @{}
    $removedByName = @{}

    foreach ($d in $dirs) {
        $prevMap = $previousMaps[$d]
        $currMap = $currentMaps[$d]
        $prevNames = @($prevMap.Keys)
        $currNames = @($currMap.Keys)

        $new = $currNames | Where-Object { $_ -notin $prevNames }
        $gone = $prevNames | Where-Object { $_ -notin $currNames }
        $common = $currNames | Where-Object { $_ -in $prevNames }

        foreach ($name in $new) {
            $addedByName[$name] = $d
        }
        foreach ($name in $gone) {
            $removedByName[$name] = $d
        }
        foreach ($name in $common) {
            $prevRow = $prevMap[$name]
            $currRow = $currMap[$name]
            if (($prevRow["LastWrite"] -ne $currRow["LastWrite"]) -or ($prevRow["Length"] -ne $currRow["Length"])) {
                $changes.Add("${d}: UPDATED -> $name")
            }
        }
    }

    foreach ($name in @($removedByName.Keys)) {
        if ($addedByName.ContainsKey($name)) {
            $fromDir = $removedByName[$name]
            $toDir = $addedByName[$name]
            $changes.Add("MOVED: $name ($fromDir -> $toDir)")
            $addedByName.Remove($name) | Out-Null
            $removedByName.Remove($name) | Out-Null
        }
    }

    foreach ($name in @($addedByName.Keys | Sort-Object)) {
        $targetDir = $addedByName[$name]
        $changes.Add("${targetDir}: NEW -> $name")
        if ($targetDir -eq "3_Done") {
            $highlights.Add("ALERT: TASK COMPLETED -> $name")
        } elseif ($targetDir -eq "4_Block") {
            $highlights.Add("ALERT: TASK BLOCKED -> $name")
        }
    }

    foreach ($name in @($removedByName.Keys | Sort-Object)) {
        $sourceDir = $removedByName[$name]
        $changes.Add("${sourceDir}: REMOVED -> $name")
    }

    $summary = @()
    foreach ($d in $dirs) {
        $count = 0
        if ($current.ContainsKey($d)) { $count = $current[$d].Count }
        $summary += "$d=$count"
    }

    $msg = "STATUS: $($summary -join ' | ')"
    Write-LogLine $msg

    if ($initializeBaseline) {
        Write-LogLine "BASELINE: snapshot initialized; change alerts will start on the next cycle."
        Write-LatestStatus -Summary $summary -Changes @()
        $current | ConvertTo-Json -Compress | Set-Content -LiteralPath $snapshotFile -Force -Encoding UTF8
        $previous = $current
        $initializeBaseline = $false
        $iteration += 1
        if ($MaxIterations -gt 0 -and $iteration -ge $MaxIterations) {
            break
        }
        Start-Sleep -Seconds $IntervalSeconds
        continue
    }

    if ($changes.Count -gt 0) {
        Write-LogLine "CHANGES DETECTED:"
        foreach ($c in $changes) { Write-LogLine "  $c" }
        foreach ($highlight in $highlights) { Write-LogLine $highlight }
    } else {
        $wakeTime = (Get-Date).AddSeconds($IntervalSeconds).ToString("yyyy-MM-dd HH:mm:ss")
        Write-LogLine "IDLE: no changes detected; sleep ${IntervalSeconds}s until $wakeTime"
    }

    Write-LatestStatus -Summary $summary -Changes @($changes)
    $current | ConvertTo-Json -Compress | Set-Content -LiteralPath $snapshotFile -Force -Encoding UTF8
    $previous = $current
    $iteration += 1

    if ($MaxIterations -gt 0 -and $iteration -ge $MaxIterations) {
        break
    }

    Start-Sleep -Seconds $IntervalSeconds
}

Write-LogLine "=== Agent1 Monitor FINISHED ==="
