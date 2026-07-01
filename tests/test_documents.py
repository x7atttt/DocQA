import io

import pytest

from app.services.document_service import get_user_collection


async def _register_and_login(client, username: str = "alice", password: str = "secret123"):
    await client.post("/api/auth/register", json={"username": username, "password": password})
    resp = await client.post("/api/auth/login", json={"username": username, "password": password})
    return resp.json()["data"]["access_token"]


def _auth_header(token: str) -> dict:
    return {"Authorization": f"Bearer {token}"}


async def _wait_document_done(client, token: str, doc_id: int, timeout: int = 30) -> str:
    """轮询文档列表，等待指定文档状态变为 done/failed（后台处理完成）。

    C2 异步上传：upload 立即返回 pending，后台 BackgroundTasks 处理。
    测试需要等后台跑完才能验证 chunk_count 等结果。
    """
    import asyncio

    for _ in range(timeout):
        await asyncio.sleep(1)
        resp = await client.get("/api/documents/list", headers=_auth_header(token))
        docs = resp.json()["data"]["documents"]
        doc = next((d for d in docs if d["id"] == doc_id), None)
        if doc and doc["status"] in ("done", "failed"):
            return doc["status"]
    return "timeout"


@pytest.mark.asyncio
async def test_upload_markdown(client):
    token = await _register_and_login(client)
    content = "# 测试文档\n\n这是一个用于 RAG 测试的文档。" * 50
    resp = await client.post(
        "/api/documents/upload",
        headers=_auth_header(token),
        files={"file": ("test.md", io.BytesIO(content.encode("utf-8")), "text/markdown")},
    )
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["code"] == 0
    assert body["data"]["filename"] == "test.md"
    assert body["data"]["file_type"] == "md"
    assert body["data"]["status"] == "pending"  # 异步：立即返回 pending

    # 轮询等待后台处理完成（pending → processing → done）
    doc_id = body["data"]["id"]
    status = await _wait_document_done(client, token, doc_id, timeout=30)
    assert status == "done", f"文档处理超时或失败，最终状态: {status}"


@pytest.mark.asyncio
async def test_upload_unsupported_type(client):
    token = await _register_and_login(client)
    resp = await client.post(
        "/api/documents/upload",
        headers=_auth_header(token),
        files={"file": ("evil.txt", io.BytesIO(b"hello"), "text/plain")},
    )
    assert resp.status_code == 400
    assert resp.json()["code"] == 20004


@pytest.mark.asyncio
async def test_upload_unauthenticated(client):
    resp = await client.post(
        "/api/documents/upload",
        files={"file": ("t.md", io.BytesIO(b"hello"), "text/markdown")},
    )
    assert resp.status_code == 401


@pytest.mark.asyncio
async def test_list_and_delete(client):
    token = await _register_and_login(client)
    for i in range(3):
        content = f"# 文档 {i}\n\n" + "内容 " * 200
        await client.post(
            "/api/documents/upload",
            headers=_auth_header(token),
            files={"file": (f"doc{i}.md", io.BytesIO(content.encode("utf-8")), "text/markdown")},
        )

    resp = await client.get("/api/documents/list", headers=_auth_header(token))
    assert resp.status_code == 200
    docs = resp.json()["data"]["documents"]
    assert len(docs) == 3

    doc_id = docs[0]["id"]
    del_resp = await client.delete(f"/api/documents/{doc_id}", headers=_auth_header(token))
    assert del_resp.status_code == 200
    assert del_resp.json()["code"] == 0

    resp2 = await client.get("/api/documents/list", headers=_auth_header(token))
    assert len(resp2.json()["data"]["documents"]) == 2


@pytest.mark.asyncio
async def test_user_isolation_chroma(client):
    token_a = await _register_and_login(client, "userA", "secret123")
    token_b = await _register_and_login(client, "userB", "secret123")

    content = "# 隔离测试\n\n" + "独占内容 " * 100
    await client.post(
        "/api/documents/upload",
        headers=_auth_header(token_a),
        files={"file": ("a.md", io.BytesIO(content.encode("utf-8")), "text/markdown")},
    )

    coll_a = get_user_collection(1)
    coll_b = get_user_collection(2)
    assert coll_a.count() > 0
    assert coll_b.count() == 0


@pytest.mark.asyncio
async def test_cursor_pagination(client):
    token = await _register_and_login(client)
    for i in range(5):
        content = f"# 文档 {i}\n\n" + "x " * 200
        await client.post(
            "/api/documents/upload",
            headers=_auth_header(token),
            files={"file": (f"d{i}.md", io.BytesIO(content.encode("utf-8")), "text/markdown")},
        )

    resp = await client.get("/api/documents/list?limit=2", headers=_auth_header(token))
    data = resp.json()["data"]
    assert len(data["documents"]) == 2
    assert data["has_next"] is True
    cursor = data["next_cursor"]

    resp2 = await client.get(
        f"/api/documents/list?limit=2&cursor={cursor}", headers=_auth_header(token)
    )
    data2 = resp2.json()["data"]
    assert len(data2["documents"]) == 2
    assert data2["documents"][0]["id"] < data["documents"][-1]["id"]


# ---------- 同名上传确认测试（增量更新）----------


@pytest.mark.asyncio
async def test_upload_same_name_same_content_rejected(client):
    """同名 + 同内容 → 拒绝（DOC_ALREADY_EXISTS，防重复）"""
    token = await _register_and_login(client)
    content = "# 原始文档\n\n" + "内容内容 " * 100
    files = {"file": ("dup.md", io.BytesIO(content.encode("utf-8")), "text/markdown")}

    resp1 = await client.post("/api/documents/upload", headers=_auth_header(token), files=files)
    assert resp1.status_code == 200

    # 第二次完全相同 → 409 DOC_ALREADY_EXISTS (20005)
    files2 = {"file": ("dup.md", io.BytesIO(content.encode("utf-8")), "text/markdown")}
    resp2 = await client.post("/api/documents/upload", headers=_auth_header(token), files=files2)
    assert resp2.status_code == 409
    assert resp2.json()["code"] == 20005


@pytest.mark.asyncio
async def test_upload_same_name_diff_content_returns_conflict(client):
    """同名 + 不同内容（不带 replace_id）→ 409 DOC_SAME_NAME_CONFLICT，返回旧文档 id"""
    token = await _register_and_login(client)

    content_v1 = "# 版本一\n\n这是原始内容。" * 50
    resp1 = await client.post(
        "/api/documents/upload",
        headers=_auth_header(token),
        files={"file": ("conflict.md", io.BytesIO(content_v1.encode("utf-8")), "text/markdown")},
    )
    assert resp1.status_code == 200
    old_id = resp1.json()["data"]["id"]

    # 同名不同内容 → 冲突（20006），带 existing_id
    content_v2 = "# 版本二\n\n内容完全不同了。" * 50
    resp2 = await client.post(
        "/api/documents/upload",
        headers=_auth_header(token),
        files={"file": ("conflict.md", io.BytesIO(content_v2.encode("utf-8")), "text/markdown")},
    )
    assert resp2.status_code == 409
    body = resp2.json()
    assert body["code"] == 20006  # DOC_SAME_NAME_CONFLICT
    assert body["data"]["existing_id"] == old_id  # 返回旧文档 id 供前端确认
    assert body["data"]["filename"] == "conflict.md"

    # 旧文档仍在（未删除，等用户确认）
    list_resp = await client.get("/api/documents/list", headers=_auth_header(token))
    assert len(list_resp.json()["data"]["documents"]) == 1


@pytest.mark.asyncio
async def test_upload_with_replace_id_updates(client):
    """带 replace_id → 增量更新：复用旧文档记录（id 不变），后台 diff chunks。

    v1→v2 内容大改（版本一→版本二）→ 变化率高 → 降级全量重建路径。
    """
    token = await _register_and_login(client)

    content_v1 = "# 版本一\n\n原始内容。" * 50
    resp1 = await client.post(
        "/api/documents/upload",
        headers=_auth_header(token),
        files={"file": ("updatable.md", io.BytesIO(content_v1.encode("utf-8")), "text/markdown")},
    )
    old_id = resp1.json()["data"]["id"]

    # 带 replace_id 上传 v2（模拟用户点了"更新"）
    content_v2 = "# 版本二\n\n全新内容。" * 50
    resp2 = await client.post(
        f"/api/documents/upload?replace_id={old_id}",
        headers=_auth_header(token),
        files={"file": ("updatable.md", io.BytesIO(content_v2.encode("utf-8")), "text/markdown")},
    )
    assert resp2.status_code == 200, resp2.text
    updated_doc = resp2.json()["data"]
    # 增量更新：复用旧文档记录，id 不变（不再删旧建新）
    assert updated_doc["id"] == old_id

    # 列表里只有 1 个同名文档（复用旧记录，无新建）
    list_resp = await client.get("/api/documents/list", headers=_auth_header(token))
    docs = [d for d in list_resp.json()["data"]["documents"] if d["filename"] == "updatable.md"]
    assert len(docs) == 1
    # 等后台处理完成后验证 chunk_count > 0（异步：上传时 pending，chunk_count 是旧值）
    status = await _wait_document_done(client, token, old_id, timeout=30)
    assert status == "done", f"更新文档处理失败: {status}"
    list_resp2 = await client.get("/api/documents/list", headers=_auth_header(token))
    updated = next(d for d in list_resp2.json()["data"]["documents"] if d["id"] == old_id)
    assert updated["chunk_count"] >= 1


@pytest.mark.asyncio
async def test_upload_replace_id_not_found(client):
    """replace_id 指向不存在的文档 → 404"""
    token = await _register_and_login(client)
    content = "# 文档\n\n内容 " * 50
    resp = await client.post(
        "/api/documents/upload?replace_id=99999",
        headers=_auth_header(token),
        files={"file": ("x.md", io.BytesIO(content.encode("utf-8")), "text/markdown")},
    )
    assert resp.status_code == 404


@pytest.mark.asyncio
async def test_upload_diff_name_creates_new(client):
    """不同名 → 正常新建（用不同内容，避免撞 hash）"""
    token = await _register_and_login(client)

    await client.post(
        "/api/documents/upload",
        headers=_auth_header(token),
        files={"file": ("a.md", io.BytesIO(("# 文档A\n\n" + "A" * 200).encode("utf-8")), "text/markdown")},
    )
    await client.post(
        "/api/documents/upload",
        headers=_auth_header(token),
        files={"file": ("b.md", io.BytesIO(("# 文档B\n\n" + "B" * 200).encode("utf-8")), "text/markdown")},
    )

    list_resp = await client.get("/api/documents/list", headers=_auth_header(token))
    assert len(list_resp.json()["data"]["documents"]) == 2
