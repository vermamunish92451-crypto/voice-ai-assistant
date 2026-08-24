# Double-click to start - auto loads .env
Set-Location $PSScriptRoot
if (Test-Path ".env") { Write-Host "✓ .env found, loading GROQ key..." -ForegroundColor Green } else { Write-Host "⚠ .env not found, using echo mode. Create .env from .env.example" -ForegroundColor Yellow }
python app_py314.py
pause
