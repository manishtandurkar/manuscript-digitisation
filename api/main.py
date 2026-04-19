from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, field_validator

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = _PROJECT_ROOT / "data" / "raw"
DATA_DIR = _PROJECT_ROOT / "data"
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".tif", ".tiff"}


class ImageMeta(BaseModel):
    id: str
    filename: str
    url: str
    thumbnail_url: str


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
                thumbnail_url=f"/api/images/{path.stem}/thumbnail",
            ))
    return images


@app.get("/api/images/{image_id}/thumbnail")
def get_thumbnail(image_id: str) -> FileResponse:
    from api.pipeline import make_thumbnail
    thumb = make_thumbnail(image_id)
    if thumb is None:
        raise HTTPException(status_code=404, detail="Image not found")
    return FileResponse(str(thumb), media_type="image/jpeg")


class ProcessRequest(BaseModel):
    image_ids: list[str]
    stages: list[str]

    @field_validator("image_ids")
    @classmethod
    def images_not_empty(cls, v: list[str]) -> list[str]:
        if not v:
            raise ValueError("image_ids must not be empty")
        return v


@app.post("/api/process")
def start_process(req: ProcessRequest) -> dict:
    from api.jobs import create_job, start_job
    job_id = create_job(req.image_ids, req.stages)
    start_job(job_id, req.image_ids, req.stages)
    return {"job_id": job_id}


@app.get("/api/jobs/{job_id}")
def get_job_status(job_id: str) -> dict:
    from api.jobs import get_job
    job = get_job(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Job not found")
    return job
