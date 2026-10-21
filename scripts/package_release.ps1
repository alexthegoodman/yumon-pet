param(
    [string]$Version = "v0.1.0"
)

$ErrorActionPreference = "Stop"

Write-Host "==========================================" -ForegroundColor Cyan
Write-Host " Building & Packaging Yumon Portal Release" -ForegroundColor Cyan
Write-Host " Version: $Version" -ForegroundColor Cyan
Write-Host "==========================================" -ForegroundColor Cyan

$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$repoDir = Split-Path -Parent $scriptDir

# 1. Build optimized release binary
Write-Host "`n[1/4] Compiling release binary (yumon_portal)..." -ForegroundColor Yellow
cargo build --release --manifest-path "$repoDir/Cargo.toml" --bin yumon_portal
if ($LASTEXITCODE -ne 0) {
    Write-Error "Release build failed."
}

# 2. Stage files
$bundleName = "YumonPortal-$Version-windows"
$distDir = Join-Path $repoDir "dist"
$stageDir = Join-Path $distDir $bundleName
$zipPath = Join-Path $distDir "$bundleName.zip"

Write-Host "`n[2/4] Staging distribution files in $stageDir ..." -ForegroundColor Yellow
if (Test-Path $stageDir) {
    Remove-Item $stageDir -Recurse -Force
}
New-Item -ItemType Directory -Force -Path $stageDir | Out-Null

Copy-Item "$repoDir/target/release/yumon_portal.exe" "$stageDir/yumon_portal.exe"

$decompressorSrc = Join-Path $repoDir "training-cache/ngram_decompressor.bin"
if (-not (Test-Path $decompressorSrc)) {
    Write-Error "Could not find $decompressorSrc"
}
Copy-Item $decompressorSrc "$stageDir/ngram_decompressor.bin"

# 3. Create zip archive
Write-Host "`n[3/4] Creating zip archive $zipPath ..." -ForegroundColor Yellow
if (Test-Path $zipPath) {
    Remove-Item $zipPath -Force
}
Compress-Archive -Path "$stageDir/*" -DestinationPath $zipPath -Force

$zipItem = Get-Item $zipPath
$zipSizeMb = [math]::Round($zipItem.Length / 1MB, 2)

Write-Host "`n[4/4] Package complete!" -ForegroundColor Green
Write-Host "Archive: $zipPath ($zipSizeMb MB)" -ForegroundColor Green

Write-Host "`nTo publish this release to GitHub:" -ForegroundColor Cyan
Write-Host "  Option A (GitHub CLI):" -ForegroundColor White
Write-Host "    gh release create $Version `"$zipPath`" --title `"Yumon Portal $Version`" --notes `"Yumon Portal $Version standalone Windows release.`"" -ForegroundColor Gray
Write-Host "  Option B (GitHub Web UI):" -ForegroundColor White
Write-Host "    1. Navigate to your repository releases on GitHub." -ForegroundColor Gray
Write-Host "    2. Click 'Draft a new release', enter tag '$Version'." -ForegroundColor Gray
Write-Host "    3. Drag and drop '$zipPath' into the attachment box." -ForegroundColor Gray
Write-Host "    4. Click 'Publish release'." -ForegroundColor Gray
