from __future__ import annotations

from fastapi.testclient import TestClient

from api.main import app

client = TestClient(app)


def test_list_images_returns_200():
    response = client.get("/api/images")
    assert response.status_code == 200


def test_list_images_returns_list():
    response = client.get("/api/images")
    data = response.json()
    assert isinstance(data, list)


def test_list_images_items_have_required_fields():
    response = client.get("/api/images")
    images = response.json()
    assert len(images) > 0, "Expected at least one image in data/raw/"
    img = images[0]
    assert "id" in img
    assert "filename" in img
    assert "url" in img


def test_list_images_urls_start_with_data():
    response = client.get("/api/images")
    images = response.json()
    for img in images:
        assert img["url"].startswith("/data/"), f"Bad URL: {img['url']}"


def test_process_returns_job_id():
    response = client.post("/api/process", json={
        "image_ids": ["IMG_3915"],
        "stages": ["preprocess"],
    })
    assert response.status_code == 200
    data = response.json()
    assert "job_id" in data
    assert isinstance(data["job_id"], str)


def test_process_rejects_empty_images():
    response = client.post("/api/process", json={
        "image_ids": [],
        "stages": ["preprocess"],
    })
    assert response.status_code == 422


def test_get_job_returns_status():
    create = client.post("/api/process", json={
        "image_ids": ["IMG_3915"],
        "stages": ["preprocess"],
    })
    job_id = create.json()["job_id"]
    response = client.get(f"/api/jobs/{job_id}")
    assert response.status_code == 200
    data = response.json()
    assert "job_id" in data
    assert "status" in data
    assert "total" in data
    assert "completed" in data
    assert "results" in data


def test_get_job_unknown_id_returns_404():
    response = client.get("/api/jobs/nonexistent-id")
    assert response.status_code == 404
