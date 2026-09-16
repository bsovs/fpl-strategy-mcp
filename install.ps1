$ErrorActionPreference = "Stop"

$repo = if ($env:FPL_STRATEGY_REPO) { $env:FPL_STRATEGY_REPO } else { "bsovs/fpl-strategy-mcp" }
$version = if ($env:FPL_STRATEGY_VERSION) { $env:FPL_STRATEGY_VERSION } else { "latest" }
$clients = if ($env:FPL_STRATEGY_CLIENTS) { $env:FPL_STRATEGY_CLIENTS } else { "none" }
if ($version -ne "latest" -and $version.StartsWith("v")) { $version = $version.Substring(1) }
$installDir = if ($env:FPL_STRATEGY_INSTALL_DIR) { $env:FPL_STRATEGY_INSTALL_DIR } else { Join-Path $env:LOCALAPPDATA "fpl-strategy\bin" }
$artifact = "fpl-strategy-mcp-windows-x64.exe"

if ($version -eq "latest") {
    $url = "https://github.com/$repo/releases/latest/download/$artifact"
} else {
    $url = "https://github.com/$repo/releases/download/v$version/$artifact"
}

New-Item -ItemType Directory -Force -Path $installDir | Out-Null
$destination = Join-Path $installDir $artifact
Write-Host "Downloading $artifact from $url"
Invoke-WebRequest -Uri $url -OutFile $destination

$userPath = [Environment]::GetEnvironmentVariable("Path", "User")
$entries = @($userPath -split ";" | Where-Object { $_ })
if ($entries -notcontains $installDir) {
    [Environment]::SetEnvironmentVariable("Path", (($entries + $installDir) -join ";"), "User")
    Write-Host "Added $installDir to the user PATH. Open a new PowerShell window."
}

Write-Host "Installed $destination"

if ($clients -and $clients -ne "none") {
    & $destination setup --clients $clients
    if ($LASTEXITCODE -ne 0) { throw "MCP client setup failed" }
}

& $destination status
if ($LASTEXITCODE -ne 0) { throw "Installed binary health check failed" }
