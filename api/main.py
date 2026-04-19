from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = _PROJECT_ROOT / "data" / "raw"
DATA_DIR = _PROJECT_ROOT / "data"
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".tif", ".tiff"}


class ImageMeta(BaseModel):
    id: str
    filename: str
    url: str


app = FastAPI(title="IDP Web UI")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],
    allow_methods=["*"],
    allow_headers=["*"],
)

if DATA_DIR.exists():
    app.mount("/data", StaticFiles(directory=str(DATA_DIR)), name="data")


@app.get("/api/images", response_model=list[ImageMeta])
def list_images() -> list[ImageMeta]:
    images = []
    for path in sorted(RAW_DIR.rglob("*")):
        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES:
            relative = path.relative_to(DATA_DIR)
            images.append(ImageMeta(
                id=path.stem,
                filename=path.name,
                url=f"/data/{relative.as_posix()}",
            ))
    return images
