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
