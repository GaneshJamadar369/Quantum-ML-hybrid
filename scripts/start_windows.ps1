param(
    [switch]$SkipBuild,
    [switch]$NoBrowser
)

$ErrorActionPreference = "Stop"
$RepoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$BundleManifest = Join-Path $RepoRoot "prototype_bundle\current\manifest.json"

Set-Location $RepoRoot

if (-not (Get-Command docker -ErrorAction SilentlyContinue)) {
    throw "Docker was not found. Install and start Docker Desktop, then run this script again."
}

docker info *> $null
if ($LASTEXITCODE -ne 0) {
    throw "Docker Desktop is installed but its engine is not running."
}

if (-not (Test-Path $BundleManifest)) {
    throw @"
The signed model bundle is missing.
Extract aquire-hybrid-q4-v1.zip so this file exists:
$BundleManifest
"@
}

if ($SkipBuild) {
    docker compose up -d
} else {
    docker compose up -d --build
}
if ($LASTEXITCODE -ne 0) {
    throw "Docker Compose could not start the AQUIRE-Med stack."
}

$ReadyUrl = "http://localhost:8080/api/v1/health/ready"
$ready = $false
for ($attempt = 1; $attempt -le 90; $attempt++) {
    try {
        $response = Invoke-RestMethod -Uri $ReadyUrl -TimeoutSec 3
        if ($response.model_ready -eq $true) {
            $ready = $true
            break
        }
    } catch {
        Start-Sleep -Seconds 2
    }
}

if (-not $ready) {
    docker compose logs --tail=80 api
    throw "The API did not become ready. Review the container log printed above."
}

Write-Host ""
Write-Host "AQUIRE-Med is ready." -ForegroundColor Green
Write-Host "Web application: http://localhost:8080"
Write-Host "API documentation: http://localhost:8000/docs"
Write-Host "MI sample: demo_samples\sample_1_mi_pattern.csv"
Write-Host "Non-MI sample: demo_samples\sample_2_non_mi.csv"
Write-Host "Stop with: docker compose down"

if (-not $NoBrowser) {
    Start-Process "http://localhost:8080"
}
