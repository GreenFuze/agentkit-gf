# Setup script for PyPI publishing
# This script helps you set up the necessary configuration for publishing to PyPI

Write-Host "🔧 Setting up PyPI publishing for agentkit-gf" -ForegroundColor Green

# Check if .pypirc already exists
$pypircPath = [System.IO.Path]::Combine($env:USERPROFILE, ".pypirc")
if (Test-Path $pypircPath) {
    Write-Host "⚠️  .pypirc already exists at $pypircPath" -ForegroundColor Yellow
    $overwrite = Read-Host "Do you want to overwrite it? (y/N)"
    if ($overwrite -ne "y" -and $overwrite -ne "Y") {
        Write-Host "❌ Setup cancelled." -ForegroundColor Red
        exit 0
    }
}

# Copy the template
Write-Host "📋 Copying .pypirc template..." -ForegroundColor Yellow
if (Test-Path ".pypirc.template") {
    Copy-Item ".pypirc.template" $pypircPath -Force
    Write-Host "✅ .pypirc template copied to $pypircPath" -ForegroundColor Green
} else {
    Write-Error "❌ .pypirc.template not found!"
    exit 1
}

# Install required tools
Write-Host "📦 Installing required tools..." -ForegroundColor Yellow
python -m pip install --upgrade build twine

# Test the build
Write-Host "🔨 Testing package build..." -ForegroundColor Yellow
python -m build --sdist --wheel --outdir test-build

if ($LASTEXITCODE -eq 0) {
    Write-Host "✅ Build test successful!" -ForegroundColor Green
    # Clean up test build
    if (Test-Path "test-build") {
        Remove-Item "test-build" -Recurse -Force
    }
} else {
    Write-Host "⚠️  Build test had issues. You may need to fix some configuration." -ForegroundColor Yellow
}

Write-Host ""
Write-Host "🎉 Setup complete!" -ForegroundColor Green
Write-Host ""
Write-Host "Next steps:" -ForegroundColor Cyan
Write-Host "1. Edit $pypircPath and add your PyPI API token" -ForegroundColor White
Write-Host "2. Get your API token from: https://pypi.org/manage/account/token/" -ForegroundColor White
Write-Host "3. Run: .\publish-to-pypi.ps1" -ForegroundColor White
Write-Host ""
Write-Host "To test first, you can:" -ForegroundColor Cyan
Write-Host "1. Use TestPyPI: https://test.pypi.org/manage/account/token/" -ForegroundColor White
Write-Host "2. Update .pypirc to use testpypi instead of pypi" -ForegroundColor White
Write-Host "3. Run: python -m twine upload --repository testpypi dist/*" -ForegroundColor White
