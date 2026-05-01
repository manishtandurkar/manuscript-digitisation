from __future__ import annotations

import copy
import threading
import uuid

from api.pipeline import run_stage


_jobs: dict[str, dict] = {}
_lock = threading.Lock()
_enhance_gate = threading.Semaphore(1)


def create_job(
    image_ids: list[str],
    stages: list[str],
    stage_options: dict[str, dict[str, str]] | None = None,
) -> str:
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
            "stage_options": stage_options or {},
        }
    return job_id


def get_job(job_id: str) -> dict | None:
    with _lock:
        job = _jobs.get(job_id)
        return copy.deepcopy(job) if job else None


def update_stage(job_id: str, image_id: str, stage: str, result: dict) -> None:
    with _lock:
        _jobs[job_id]["results"][image_id][stage] = result


def mark_job_failed(job_id: str) -> None:
    with _lock:
        if job_id in _jobs:
            _jobs[job_id]["status"] = "failed"


def mark_image_done(job_id: str) -> None:
    with _lock:
        _jobs[job_id]["completed"] += 1
        if _jobs[job_id]["completed"] >= _jobs[job_id]["total"]:
            _jobs[job_id]["status"] = "done"


def _process_image(
    job_id: str,
    image_id: str,
    stages: list[str],
    stage_options: dict[str, dict[str, str]],
) -> None:
    current_stage: str | None = None
    try:
        for stage in stages:
            current_stage = stage
            update_stage(job_id, image_id, stage, {"status": "running"})
            options = stage_options.get(stage, {})

            if stage == "enhance":
                with _enhance_gate:
                    result = run_stage(image_id, stage, options)
            else:
                result = run_stage(image_id, stage, options)

            update_stage(job_id, image_id, stage, result)

        mark_image_done(job_id)
    except Exception as exc:
        # Guard against unexpected thread failures so jobs never stay "running" forever.
        if current_stage is not None:
            update_stage(job_id, image_id, current_stage, {"status": "failed", "error": str(exc)})
        mark_job_failed(job_id)


def start_job(
    job_id: str,
    image_ids: list[str],
    stages: list[str],
    stage_options: dict[str, dict[str, str]] | None = None,
) -> None:
    opts = stage_options or {}
    for image_id in image_ids:
        t = threading.Thread(
            target=_process_image,
            args=(job_id, image_id, stages, opts),
            daemon=True,
        )
        t.start()
