# Run AQUIRE-Med on a Windows PC

## What to share

The recipient needs:

1. The public GitHub repository:
   `https://github.com/GaneshJamadar369/Quantum-ML-hybrid.git`
2. The separately supplied `aquire-hybrid-q4-v1.zip` model bundle and its
   `.sha256` checksum file.

The model bundle is intentionally excluded from Git because it contains frozen
weights and patient-derived transformation artifacts.

## Requirements

- Windows 10 or 11, 64 bit
- Docker Desktop using the WSL 2 backend
- Git for Windows
- At least 8 GB RAM and roughly 5 GB of free disk space for the first build

Docker Desktop must be running before starting the platform. No local Python,
Node.js or quantum SDK installation is required.

## First-time setup

Open PowerShell and run:

```powershell
git clone https://github.com/GaneshJamadar369/Quantum-ML-hybrid.git
cd Quantum-ML-hybrid
New-Item -ItemType Directory -Force .\prototype_bundle\current | Out-Null
Expand-Archive -Path "$HOME\Downloads\aquire-hybrid-q4-v1.zip" `
  -DestinationPath .\prototype_bundle\current -Force
```

Confirm that this file exists:

```text
prototype_bundle\current\manifest.json
```

Optionally verify the downloaded ZIP before extracting it:

```powershell
Get-FileHash "$HOME\Downloads\aquire-hybrid-q4-v1.zip" -Algorithm SHA256
Get-Content "$HOME\Downloads\aquire-hybrid-q4-v1.zip.sha256"
```

The two hashes must be identical.

## Start the platform

From the repository directory:

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\start_windows.ps1
```

The first build downloads the pinned scientific dependencies and can take
several minutes. Later starts reuse the Docker image and are much faster.

When the signed golden self-test passes, the script opens:

- Web application: `http://localhost:8080`
- API documentation: `http://localhost:8000/docs`

Upload either example from the `demo_samples` directory. Both the quantum VQC
and classical HGB routes must show as active in the result.

## Later starts and stops

```powershell
# Start using the existing images
powershell -ExecutionPolicy Bypass -File .\scripts\start_windows.ps1 -SkipBuild

# Stop the platform
docker compose down

# Inspect errors
docker compose logs --tail=100 api web
```

## Common problems

- **Port 8000 or 8080 is already in use:** stop the application using that
  port, then rerun the launcher.
- **Model not ready:** confirm that the ZIP contents were extracted directly
  into `prototype_bundle\current`, rather than into an additional nested
  folder.
- **Docker engine not running:** open Docker Desktop and wait until it reports
  that the engine is ready.
- **Slow first build:** the CPU Torch image and scientific Python wheels are
  downloaded only once and then cached by Docker Desktop.

This prototype processes uploads locally. It produces a research MI-pattern
screening result, not a diagnosis or future cardiovascular-risk estimate.
