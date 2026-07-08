$ErrorActionPreference = "Stop"

try {
    $destRoot = (git rev-parse --show-toplevel 2>$null).Trim()
    if (-not $destRoot) {
        exit 0
    }

    $destRoot = [System.IO.Path]::GetFullPath($destRoot)
    $destLogs = Join-Path $destRoot "CC-Session-Logs"

    $sourceLogs = $env:CXR_MC_SESSION_LOG_SOURCE
    if ($sourceLogs) {
        $sourceLogs = [System.IO.Path]::GetFullPath($sourceLogs)
    } else {
        $records = @()
        $record = @{}
        foreach ($line in (git worktree list --porcelain)) {
            if (-not $line) {
                if ($record.Count -gt 0) {
                    $records += [pscustomobject]$record
                    $record = @{}
                }
                continue
            }

            $space = $line.IndexOf(" ")
            if ($space -lt 0) {
                continue
            }

            $key = $line.Substring(0, $space)
            $value = $line.Substring($space + 1)
            $record[$key] = $value
        }
        if ($record.Count -gt 0) {
            $records += [pscustomobject]$record
        }

        $main = $records |
            Where-Object { $_.branch -eq "refs/heads/main" } |
            Select-Object -First 1
        if (-not $main) {
            exit 0
        }

        $sourceLogs = Join-Path $main.worktree "CC-Session-Logs"
        $sourceLogs = [System.IO.Path]::GetFullPath($sourceLogs)
    }

    if (-not (Test-Path -LiteralPath $sourceLogs -PathType Container)) {
        exit 0
    }

    if ($sourceLogs.TrimEnd("\") -ieq $destLogs.TrimEnd("\")) {
        exit 0
    }

    New-Item -ItemType Directory -Force -Path $destLogs | Out-Null
    Get-ChildItem -Force -LiteralPath $sourceLogs |
        Copy-Item -Destination $destLogs -Recurse -Force
} catch {
    Write-Warning "Unable to sync CC-Session-Logs: $($_.Exception.Message)"
    exit 0
}
