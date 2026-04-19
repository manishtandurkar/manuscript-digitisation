from __future__ import annotations

import copy
import threading
import uuid

from api.pipeline import run_stage


_jobs: dict[str, dict] = {}
_lock = threading.Lock()


def create_job(image_ids: list[str], stages: list[str]) -> str:
    job_id = str(uuid.uuid4())
    results = {
        img_id: {stage: {"status": "pending"} for stage in stages}
        for img_id in image_ids
    }
    with _lock:
        _jobs[job_id] = {
            "job_id": job_id,
            "status": "running",
            "total": len(image_ids),
            "completed": 0,
            "results": results,
        }
    return job_id


def get_job(job_id: str) -> dict | None:
    with _lock:
        job = _jobs.get(job_id)
        return copy.deepcopy(job) if job else None


def update_stage(job_id: str, image_id: str, stage: str, result: dict) -> None:
    with _lock:
        _jobs[job_id]["results"][image_id][stage] = result


def mark_image_done(job_id: str) -> None:
    with _lock:
        _jobs[job_id]["completed"] += 1
        if _jobs[job_id]["completed"] >= _jobs[job_id]["total"]:
            _jobs[job_id]["status"] = "done"


def _process_image(job_id: str, image_id: str, stages: list[str]) -> None:
    for stage in stages:
        result = run_stage(image_id, stage)
        update_stage(job_id, image_id, stage, result)
    mark_image_done(job_id)


def start_job(job_id: str, image_ids: list[str], stages: list[str]) -> None:
    for image_id in image_ids:
        t = threading.Thread(
            target=_process_image,
            args=(job_id, image_id, stages),
            daemon=True,
        )
        t.start()
