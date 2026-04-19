from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

RAW_DIR = Path("data/raw")
DATA_DIR = Path("data")
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".tif", ".tiff"}

app = FastAPI(title="IDP Web UI")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/data", StaticFiles(directory=str(DATA_DIR)), name="data")


@app.get("/api/images")
def list_images() -> list[dict]:
    images = []
    for path in sorted(RAW_DIR.rglob("*")):
        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES:
            relative = path.relative_to(DATA_DIR)
            images.append({
                "id": path.stem,
                "filename": path.name,
                "url": f"/data/{relative.as_posix()}",
            })
    return images
