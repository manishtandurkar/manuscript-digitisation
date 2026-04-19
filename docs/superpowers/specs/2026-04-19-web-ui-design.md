# Web UI Design — Inscription Digitisation Pipeline

**Date:** 2026-04-19  
**Status:** Approved

## Overview

Replace CLI-driven pipeline execution with a browser-based web UI. Users can browse all raw images, select individual or multiple images, choose which pipeline stages to run, and view processed outputs side-by-side with the originals — without touching the terminal.

## Architecture

```
project/
├── src/                        (existing pipeline code — unchanged)
├── data/                       (existing image dirs — unchanged)
│   ├── raw/                    (154 input images)
│   ├── enhanced/
│   ├── binarised/
│   ├── transcriptions/
│   └── translations/
├── api/
│   └── main.py                 (FastAPI server)
└── web/
    ├── src/
    │   ├── components/
    │   │   ├── ImageGrid.tsx
    │   │   ├── ImageCard.tsx
    │   │   ├── StagePanel.tsx
    │   │   └── ResultViewer.tsx
    │   ├── hooks/
    │   │   ├── useImages.ts
    │   │   └── useJob.ts
    │   └── App.tsx
    ├── package.json
    └── vite.config.ts          (proxies /api/* → :8000)
```

## Backend — FastAPI (`api/main.py`)

### Endpoints

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/api/images` | List all images in `data/raw/` with metadata |
| `GET` | `/api/images/{id}/thumbnail` | Serve image file (raw or processed) |
| `POST` | `/api/process` | Start a processing job |
| `GET` | `/api/jobs/{job_id}` | Poll job status and per-stage results |

### `/api/process` Request Body
```json
{
  "image_ids": ["img_001", "img_002"],
  "stages": ["preprocess", "enhance", "binarise", "ocr", "translate"]
}
```

### `/api/jobs/{job_id}` Response
```json
{
  "job_id": "abc123",
  "status": "running",
  "total": 2,
  "completed": 1,
  "results": {
    "img_001": {
      "preprocess": { "status": "done", "output_path": "data/enhanced/img_001.jpg" },
      "enhance":    { "status": "running" },
      "binarise":   { "status": "pending" }
    }
  }
}
```

### Processing Model
- Jobs run in `concurrent.futures.ProcessPoolExecutor` — non-blocking, parallel per image
- Each job stored in an in-memory dict (keyed by UUID job_id)
- Stage results written to existing output dirs (`data/enhanced/`, etc.)
- `data/` mounted as FastAPI `StaticFiles` so output images are served directly

## Frontend — React + Vite

### Tech Stack
- React 19 + TypeScript
- Vite 6 (dev server proxies `/api/*` to FastAPI `:8000`)
- TanStack Query v5 (server state, job polling)
- Tailwind CSS v4 (styling)

### Views

**Gallery View**
- Responsive grid of all 154 raw images
- Each card: thumbnail, filename, checkbox for selection
- Search/filter bar (by filename)
- "Select all" / "Deselect all" toggles
- Selection count badge

**Process Panel** (appears when ≥1 image selected)
- Stage checkboxes: Preprocess, Enhance (Super-resolution), Binarise, OCR, Translate
- "Run Selected Stages" button
- "Run Full Pipeline" button (all stages)
- Batch mode auto-detected when >1 image selected (no separate toggle needed)

**Results View**
- Triggered after job starts
- Per-image row: original on the left, each completed stage output as a column to the right
- Stage outputs appear as they complete (poll every 2s, render incrementally)
- OCR text and translation rendered as text below the image row
- Progress bar showing overall job completion (X of N images done)

### Error Handling
- Failed stage: inline error badge on that image's result card
- Other images/stages unaffected — partial success is valid
- Retry button per failed stage on a given image
- Network errors surfaced via toast notification

## Data Flow

1. App mounts → `GET /api/images` → renders gallery grid
2. User selects images + stages → `POST /api/process` → receives `job_id`
3. TanStack Query polls `GET /api/jobs/{job_id}` every 2s while `status !== "done"`
4. As each stage result arrives, `ResultViewer` renders the output image immediately
5. On `status === "done"` or `status === "failed"`, polling stops

## Out of Scope
- Authentication / multi-user sessions
- Persistent job history across server restarts (in-memory only)
- WebSocket streaming (polling every 2s is sufficient for this workload)
- Deployment / Docker (local dev only)
