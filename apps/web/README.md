# AQUIRE-Med web interface

Judge-facing React/Vite interface for the fixed hybrid q4-VQC plus morphology-
HGB prototype.

```bash
npm install
npm run dev
```

Set `VITE_API_BASE` when the API is not available at `http://localhost:8000`.
The interface can display architecture and benchmark content before a model
bundle is mounted, but it keeps the prediction action disabled until the API
reports a verified dual-route bundle.

Checks:

```bash
npm run build
npm run lint
```
