# PowerShell script to build and publish agentkit-gf to PyPI
# Usage: .\publish-to-pypi.ps1 [version]

param(
    [string]$Version = "0.1.3"
)

Write-Host "🚀 Publishing agentkit-gf version $Version to PyPI" -ForegroundColor Green

# Check if we're in the right directory
if (-not (Test-Path "pyproject.toml")) {
    Write-Error "❌ pyproject.toml not found. Please run this script from the project root directory."
    exit 1
}

# Check if agentkit_gf directory exists
if (-not (Test-Path "agentkit_gf")) {
    Write-Error "❌ agentkit_gf directory not found. Please run this script from the project root directory."
    exit 1
}

# Update version in pyproject.toml if provided
if ($Version -ne "0.1.3") {
    Write-Host "📝 Updating version to $Version in pyproject.toml..." -ForegroundColor Yellow
    $content = Get-Content "pyproject.toml" -Raw
    $content = $content -replace 'version = "0\.1\.0"', "version = `"$Version`""
    Set-Content "pyproject.toml" $content
}

# Check if build tools are installed
Write-Host "🔧 Checking build tools..." -ForegroundColor Yellow
try {
    python -c "import build" 2>$null
    if ($LASTEXITCODE -ne 0) {
        Write-Host "📦 Installing build tools..." -ForegroundColor Yellow
        python -m pip install --upgrade build twine
    }
} catch {
    Write-Host "📦 Installing build tools..." -ForegroundColor Yellow
    python -m pip install --upgrade build twine
}

# Clean previous builds
Write-Host "🧹 Cleaning previous builds..." -ForegroundColor Yellow
if (Test-Path "dist") {
    Remove-Item "dist" -Recurse -Force
}
if (Test-Path "build") {
    Remove-Item "build" -Recurse -Force
}
Get-ChildItem -Name "*.egg-info" -Directory | ForEach-Object {
    Remove-Item $_ -Recurse -Force
}

# Build the package
Write-Host "🔨 Building package..." -ForegroundColor Yellow
python -m build

if ($LASTEXITCODE -ne 0) {
    Write-Error "❌ Build failed!"
    exit 1
}

# Check the built package
Write-Host "🔍 Checking built package..." -ForegroundColor Yellow
python -m twine check dist/*

if ($LASTEXITCODE -ne 0) {
    Write-Error "❌ Package check failed!"
    exit 1
}

# Show what will be uploaded
Write-Host "📋 Files to be uploaded:" -ForegroundColor Cyan
Get-ChildItem "dist" | ForEach-Object {
    Write-Host "  - $($_.Name) ($([math]::Round($_.Length / 1KB, 2)) KB)" -ForegroundColor White
}

# Ask for confirmation
Write-Host ""
$confirmation = Read-Host "🤔 Do you want to upload to PyPI? (y/N)"
if ($confirmation -ne "y" -and $confirmation -ne "Y") {
    Write-Host "❌ Upload cancelled." -ForegroundColor Red
    exit 0
}

# Upload to PyPI
Write-Host "📤 Uploading to PyPI..." -ForegroundColor Green
python -m twine upload dist/*

if ($LASTEXITCODE -eq 0) {
    Write-Host "✅ Successfully published agentkit-gf version $Version to PyPI!" -ForegroundColor Green
    Write-Host "🔗 Package URL: https://pypi.org/project/agentkit-gf/" -ForegroundColor Cyan
    Write-Host "📦 Install with: pip install agentkit-gf==$Version" -ForegroundColor Cyan
} else {
    Write-Error "❌ Upload failed!"
    exit 1
}

# Clean up
Write-Host "🧹 Cleaning up build artifacts..." -ForegroundColor Yellow
if (Test-Path "dist") {
    Remove-Item "dist" -Recurse -Force
}
if (Test-Path "build") {
    Remove-Item "build" -Recurse -Force
}
Get-ChildItem -Name "*.egg-info" -Directory | ForEach-Object {
    Remove-Item $_ -Recurse -Force
}

Write-Host "✨ Done!" -ForegroundColor Green
