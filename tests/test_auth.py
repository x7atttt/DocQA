import pytest


@pytest.mark.asyncio
async def test_register_success(client):
    resp = await client.post("/api/auth/register", json={"username": "alice", "password": "secret123"})
    assert resp.status_code == 200
    body = resp.json()
    assert body["code"] == 0
    assert body["data"]["username"] == "alice"
    assert "id" in body["data"]


@pytest.mark.asyncio
async def test_register_duplicate(client):
    payload = {"username": "bob", "password": "secret123"}
    r1 = await client.post("/api/auth/register", json=payload)
    assert r1.status_code == 200
    r2 = await client.post("/api/auth/register", json=payload)
    assert r2.status_code == 409
    assert r2.json()["code"] == 10004


@pytest.mark.asyncio
async def test_register_validation_short_password(client):
    resp = await client.post("/api/auth/register", json={"username": "charlie", "password": "abc"})
    assert resp.status_code == 422
    assert resp.json()["code"] == 40001


@pytest.mark.asyncio
async def test_login_success(client):
    await client.post("/api/auth/register", json={"username": "dave", "password": "secret123"})
    resp = await client.post("/api/auth/login", json={"username": "dave", "password": "secret123"})
    assert resp.status_code == 200
    data = resp.json()["data"]
    assert data["access_token"]
    assert data["token_type"] == "bearer"
    assert data["user"]["username"] == "dave"


@pytest.mark.asyncio
async def test_login_wrong_password(client):
    await client.post("/api/auth/register", json={"username": "eve", "password": "secret123"})
    resp = await client.post("/api/auth/login", json={"username": "eve", "password": "wrong"})
    assert resp.status_code == 401
    assert resp.json()["code"] == 10005


@pytest.mark.asyncio
async def test_login_user_not_found(client):
    resp = await client.post(
        "/api/auth/login", json={"username": "ghost", "password": "secret123"}
    )
    assert resp.status_code == 401
    assert resp.json()["code"] == 10005
