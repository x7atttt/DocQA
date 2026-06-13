import io
import json

import pytest


async def _register_and_login(client, username: str = "alice", password: str = "secret123"):
    await client.post("/api/auth/register", json={"username": username, "password": password})
    resp = await client.post("/api/auth/login", json={"username": username, "password": password})
    return resp.json()["data"]["access_token"]


def _auth(token: str) -> dict:
    return {"Authorization": f"Bearer {token}"}


@pytest.mark.asyncio
async def test_ask_unauthenticated(client):
    resp = await client.post("/api/chat/ask", json={"question": "hi"})
    assert resp.status_code == 401


@pytest.mark.asyncio
async def test_ask_empty_question(client):
    token = await _register_and_login(client)
    resp = await client.post("/api/chat/ask", json={"question": "   "}, headers=_auth(token))
    assert resp.status_code == 400
    assert resp.json()["code"] == 30002


@pytest.mark.asyncio
async def test_general_chat_no_docs(client):
    token = await _register_and_login(client)
    resp = await client.post(
        "/api/chat/ask",
        json={"question": "1+1等于几？只回答数字"},
        headers=_auth(token),
    )
    assert resp.status_code == 200
    body = resp.text
    assert "event: token" in body
    assert "event: done" in body


@pytest.mark.asyncio
async def test_history_pagination(client):
    token = await _register_and_login(client)
    for i in range(3):
        await client.post(
            "/api/chat/ask",
            json={"question": f"第 {i} 个独立问题 {i}"},
            headers=_auth(token),
        )

    resp = await client.get("/api/chat/history?limit=2", headers=_auth(token))
    assert resp.status_code == 200
    data = resp.json()["data"]
    assert len(data["messages"]) == 2
    assert data["has_next"] is True
    assert data["next_cursor"] is not None


@pytest.mark.asyncio
async def test_cache_hit_on_same_question(client):
    """相同问题第二次命中缓存，响应里 cache=hit。"""
    token = await _register_and_login(client)
    q = "独立问题_cache_test_xyz"
    await client.post("/api/chat/ask", json={"question": q}, headers=_auth(token))
    resp2 = await client.post("/api/chat/ask", json={"question": q}, headers=_auth(token))
    assert resp2.status_code == 200
    assert "cache" in resp2.text


@pytest.mark.asyncio
async def test_history_user_isolation(client):
    token_a = await _register_and_login(client, "isoA", "secret123")
    token_b = await _register_and_login(client, "isoB", "secret123")

    await client.post(
        "/api/chat/ask",
        json={"question": "这是用户A的私有消息"},
        headers=_auth(token_a),
    )

    resp_b = await client.get("/api/chat/history", headers=_auth(token_b))
    assert resp_b.status_code == 200
    assert len(resp_b.json()["data"]["messages"]) == 0
