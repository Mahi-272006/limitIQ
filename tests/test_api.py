from fastapi.testclient import TestClient
from api.main import app

client = TestClient(app)


def test_health():
    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.json() == {"status": "ok"}


def test_login():
    resp = client.post("/token", data={"username": "admin", "password": "admin123"})
    assert resp.status_code == 200
    assert "access_token" in resp.json()