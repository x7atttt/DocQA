/**
 * 文档管理页：上传（拖拽+点击）、游标分页列表、删除
 */
(function () {
    "use strict";
    if (!window.API.requireAuth()) return;

    const { Token, ENDPOINTS, fetchJSON, formatSize, formatTime } = window.API;

    // 顶部用户名
    const user = Token.getUser();
    if (user) document.getElementById("navUser").textContent = `👤 ${user.username}`;

    document.getElementById("logoutBtn").addEventListener("click", () => {
        Token.clear();
        location.href = "/login.html";
    });

    // ---------- Toast ----------
    const toastEl = document.getElementById("toast");
    const toast = new bootstrap.Toast(toastEl, { delay: 2500 });
    function showToast(msg, type = "primary") {
        toastEl.className = `toast align-items-center text-bg-${type} border-0`;
        document.getElementById("toastBody").textContent = msg;
        toast.show();
    }

    // ---------- 列表 + 游标分页 ----------
    const tbody = document.getElementById("docTbody");
    const loadMoreBtn = document.getElementById("loadMoreBtn");
    const listMeta = document.getElementById("listMeta");
    let nextCursor = null;
    let hasMore = false;

    function rowHtml(doc) {
        return `<tr data-id="${doc.id}">
            <td><i class="bi ${fileIcon(doc.file_type)} me-2 text-muted"></i>${escapeHtml(doc.filename)}</td>
            <td><span class="badge bg-secondary">${escapeHtml(doc.file_type || "-")}</span></td>
            <td class="text-end">${doc.chunk_count ?? 0}</td>
            <td class="text-end">${formatSize(doc.file_size)}</td>
            <td class="small text-muted">${formatTime(doc.created_at)}</td>
            <td class="text-end">
                <button class="btn btn-sm btn-outline-danger del-btn"><i class="bi bi-trash"></i></button>
            </td>
        </tr>`;
    }

    function fileIcon(type) {
        if (type === "pdf") return "bi-file-earmark-pdf";
        if (type === "docx") return "bi-file-earmark-word";
        if (type === "md") return "bi-file-earmark-text";
        return "bi-file-earmark";
    }

    function escapeHtml(s) {
        return window.API.escapeHtml(s || "");
    }

    /** 上传成功后增量插入到列表顶部，不刷新全量（避免列表闪烁）*/
    function insertDocToTop(doc) {
        // 如果当前是空状态（"还没有文档"），清掉占位
        const placeholder = tbody.querySelector(".text-center.text-muted");
        if (placeholder) placeholder.closest("tr").remove();
        // 插到最前面
        tbody.insertAdjacentHTML("afterbegin", rowHtml(doc));
        // 更新计数
        const cnt = tbody.children.length;
        listMeta.textContent = cnt ? `共加载 ${cnt} 条` : "";
    }

    async function loadFirst() {
        tbody.innerHTML = `<tr><td colspan="6" class="text-center text-muted py-4">加载中...</td></tr>`;
        nextCursor = null;
        hasMore = false;
        await loadMore();
    }

    async function loadMore() {
        loadMoreBtn.classList.add("d-none");
        const isRefresh = nextCursor === null; // loadFirst 触发：需清空旧内容
        try {
            const params = new URLSearchParams({ limit: "20" });
            if (nextCursor) params.set("cursor", nextCursor);
            const data = await fetchJSON(`${ENDPOINTS.documents.list}?${params}`);
            const docs = data.documents || [];
            if (!nextCursor && docs.length === 0) {
                tbody.innerHTML = `<tr><td colspan="6" class="text-center text-muted py-5">
                    <i class="bi bi-inbox fs-1 d-block mb-2"></i>还没有文档，上传一个试试吧
                </td></tr>`;
            } else {
                if (isRefresh) tbody.innerHTML = ""; // 清掉"加载中..."占位
                tbody.insertAdjacentHTML("beforeend", docs.map(rowHtml).join(""));
            }
            nextCursor = data.next_cursor;
            hasMore = data.has_next;
            listMeta.textContent = docs.length ? `共加载 ${tbody.children.length} 条` : "";
            if (hasMore) loadMoreBtn.classList.remove("d-none");
        } catch (err) {
            tbody.innerHTML = `<tr><td colspan="6" class="text-center text-danger py-4">${escapeHtml(err.message)}</td></tr>`;
        }
    }

    loadFirst();
    loadMoreBtn.addEventListener("click", loadMore);

    // ---------- 删除 ----------
    tbody.addEventListener("click", async (e) => {
        const btn = e.target.closest(".del-btn");
        if (!btn) return;
        const tr = btn.closest("tr");
        const id = tr.dataset.id;
        if (!confirm("确定删除该文档？相关向量也会一并清除。")) return;
        btn.disabled = true;
        try {
            await fetchJSON(ENDPOINTS.documents.delete(id), { method: "DELETE" });
            tr.remove();
            if (!tbody.children.length) loadFirst();
            showToast("删除成功", "success");
        } catch (err) {
            showToast(err.message, "danger");
            btn.disabled = false;
        }
    });

    // ---------- 上传（拖拽 + 点击）----------
    const dropZone = document.getElementById("dropZone");
    const fileInput = document.getElementById("fileInput");
    const uploadBox = document.getElementById("uploadProgress");
    const uploadFileName = document.getElementById("uploadFileName");
    const uploadStatus = document.getElementById("uploadStatus");
    let currentXhr = null; // 保存上传 XHR 引用，beforeunload 时可中断

    dropZone.addEventListener("click", () => fileInput.click());
    ["dragover", "dragenter"].forEach((ev) =>
        dropZone.addEventListener(ev, (e) => {
            e.preventDefault();
            dropZone.classList.add("drag-over");
        })
    );
    ["dragleave", "drop"].forEach((ev) =>
        dropZone.addEventListener(ev, (e) => {
            e.preventDefault();
            dropZone.classList.remove("drag-over");
        })
    );
    dropZone.addEventListener("drop", (e) => {
        const files = e.dataTransfer.files;
        if (files && files.length) uploadFiles(files);
    });
    fileInput.addEventListener("change", () => {
        if (fileInput.files.length) uploadFiles(fileInput.files);
        fileInput.value = ""; // 允许重复选
    });

    const uploadBar = document.getElementById("uploadBar");
    const batchProgressEl = document.getElementById("batchProgress");

    // ---------- 批量上传：限并发队列 + 冲突收集 + 状态轮询 ----------

    /** 限并发上传：同时最多 CONCURRENCY 个文件在传 */
    const CONCURRENCY = 3;
    let pollTimer = null; // 状态轮询定时器

    async function uploadFiles(fileList) {
        const files = Array.from(fileList);
        // 前端预校验：过滤不支持的格式和超大文件
        const valid = [];
        for (const f of files) {
            const ext = f.name.split(".").pop().toLowerCase();
            if (!["pdf", "docx", "md"].includes(ext)) {
                showToast(`${f.name}：仅支持 PDF/DOCX/MD，已跳过`, "warning");
                continue;
            }
            if (f.size > 20 * 1024 * 1024) {
                showToast(`${f.name}：超过 20MB 限制，已跳过`, "warning");
                continue;
            }
            valid.push(f);
        }
        if (!valid.length) return;

        // 显示批量进度
        uploadBox.classList.remove("d-none");
        batchProgressEl.classList.remove("d-none");
        let completed = 0;
        const pendingConflicts = []; // 收集同名冲突

        const updateBatchProgress = () => {
            batchProgressEl.textContent = `批量上传 ${completed}/${valid.length}` +
                (pendingConflicts.length ? `，${pendingConflicts.length} 个同名待确认` : "");
        };
        updateBatchProgress();

        // 限并发队列：递归取下一个文件上传
        async function worker() {
            while (valid.length) {
                const file = valid.shift();
                await uploadOne(file, (result) => {
                    if (result && result.code === 20006) {
                        pendingConflicts.push({ file, existing_id: result.existing_id, filename: result.filename });
                    }
                    completed++;
                    updateBatchProgress();
                });
            }
        }
        // 启动 CONCURRENCY 个 worker
        await Promise.all(Array.from({ length: Math.min(CONCURRENCY, valid.length) }, () => worker()));

        // 全部上传完：如果有 pending 文档，启动状态轮询
        startStatusPolling();

        // 如果有同名冲突，弹批量确认
        if (pendingConflicts.length) {
            updateBatchProgress();
            showBatchConflictModal(pendingConflicts);
        } else if (completed === files.length) {
            uploadStatus.textContent = "全部完成";
            uploadStatus.className = "text-success";
            showToast(`批量上传完成（${completed} 个）`, "success");
        }
    }

    /** 上传单个文件（返回 Promise，resolve code/existing_id 或 null） */
    function uploadOne(file, onDone, replaceId = null) {
        return new Promise((resolve) => {
            uploadFileName.textContent = file.name;
            uploadStatus.textContent = replaceId ? "更新中..." : "上传中...";
            uploadStatus.className = "text-muted";
            uploadBar.style.width = "0%";
            uploadBar.className = "progress-bar";

            const token = Token.get();
            const fd = new FormData();
            fd.append("file", file);
            const xhr = new XMLHttpRequest();
            currentXhr = xhr;
            const url = replaceId
                ? `${ENDPOINTS.documents.upload}?replace_id=${replaceId}`
                : ENDPOINTS.documents.upload;
            xhr.open("POST", url);
            xhr.timeout = 120000;
            if (token) xhr.setRequestHeader("Authorization", `Bearer ${token}`);

            xhr.upload.onprogress = (e) => {
                if (e.lengthComputable) {
                    const pct = Math.round((e.loaded / e.total) * 100);
                    uploadBar.style.width = pct + "%";
                    if (pct < 100) uploadStatus.textContent = `上传中 ${pct}%`;
                    else {
                        uploadStatus.textContent = "后台处理中...";
                        uploadBar.classList.add("progress-bar-striped", "progress-bar-animated");
                    }
                }
            };

            xhr.onload = () => {
                if (xhr.status === 401) {
                    Token.clear();
                    location.href = "/login.html";
                    resolve(null);
                    return;
                }
                const payload = JSON.parse(xhr.responseText || "{}");
                let result = null;
                if (xhr.status === 200 && payload.code === 0) {
                    uploadStatus.textContent = "已接收";
                    uploadStatus.className = "text-success";
                    insertDocToTop(payload.data); // pending 文档插入列表
                    result = { code: 0, data: payload.data };
                } else if (payload.code === 20005) {
                    uploadStatus.textContent = "已存在";
                    uploadStatus.className = "text-warning";
                    result = { code: 20005 };
                } else if (payload.code === 20006) {
                    uploadStatus.textContent = "同名冲突";
                    uploadStatus.className = "text-warning";
                    result = { code: 20006, existing_id: payload.data.existing_id, filename: payload.data.filename };
                } else {
                    uploadStatus.textContent = "失败";
                    uploadStatus.className = "text-danger";
                    showToast(payload?.message || `${file.name} 上传失败`, "danger");
                    result = { code: -1 };
                }
                if (onDone) onDone(result);
                resolve(result);
            };

            xhr.onerror = () => {
                showToast(`${file.name} 网络错误`, "danger");
                if (onDone) onDone({ code: -1 });
                resolve({ code: -1 });
            };
            xhr.ontimeout = () => {
                showToast(`${file.name} 上传超时`, "danger");
                if (onDone) onDone({ code: -1 });
                resolve({ code: -1 });
            };
            xhr.send(fd);
        });
    }

    // ---------- 状态轮询：pending/processing → done ----------
    function startStatusPolling() {
        if (pollTimer) return; // 已在轮询
        pollTimer = setInterval(async () => {
            try {
                const data = await fetchJSON(`${ENDPOINTS.documents.list}?limit=100`);
                const docs = data.documents || [];
                const pending = docs.filter((d) => d.status === "pending" || d.status === "processing");
                if (pending.length === 0) {
                    // 全部处理完，停止轮询 + 刷新列表
                    clearInterval(pollTimer);
                    pollTimer = null;
                    await loadFirst();
                    const failed = docs.filter((d) => d.status === "failed");
                    if (failed.length) {
                        showToast(`${failed.length} 个文档处理失败`, "warning");
                    } else {
                        showToast("文档处理完成", "success");
                    }
                    // 上传进度区淡出
                    setTimeout(() => {
                        uploadBox.classList.add("d-none");
                        batchProgressEl.classList.add("d-none");
                    }, 2000);
                }
            } catch (err) {
                // 轮询失败不中断，下次继续
            }
        }, 3000); // 每 3 秒轮询一次
    }

    // ---------- 同名冲突批量确认 ----------
    const updateModal = new bootstrap.Modal(document.getElementById("updateConfirmModal"));
    let pendingFile = null;
    let pendingReplaceId = null;

    function showUpdateConfirm(file, existingId, filename) {
        pendingFile = file;
        pendingReplaceId = existingId;
        document.getElementById("conflictFilename").textContent = filename;
        updateModal.show();
    }

    // 批量冲突：逐个带 replace_id 重传（简单版：逐个确认）
    async function showBatchConflictModal(conflicts) {
        // 复用现有 modal，逐个处理冲突
        for (const c of conflicts) {
            pendingFile = c.file;
            pendingReplaceId = c.existing_id;
            document.getElementById("conflictFilename").textContent = c.filename;
            updateModal.show();
            // 等用户操作（modal hide 后继续下一个）
            await new Promise((resolve) => {
                document.getElementById("updateConfirmModal").addEventListener("hidden.bs.modal", resolve, { once: true });
            });
        }
    }

    // 用户点"更新" → 带 replace_id 重传
    document.getElementById("confirmUpdateBtn").addEventListener("click", () => {
        updateModal.hide();
        if (pendingFile && pendingReplaceId) {
            const file = pendingFile;
            const replaceId = pendingReplaceId;
            pendingFile = null;
            pendingReplaceId = null;
            uploadOne(file, (result) => {
                if (result && result.code === 0) {
                    // 更新成功：移除旧行，插入新 pending 行，启动轮询
                    const oldRow = tbody.querySelector(`tr[data-id="${replaceId}"]`);
                    if (oldRow) oldRow.remove();
                    insertDocToTop(result.data);
                    startStatusPolling();
                    showToast("文档更新已提交", "success");
                }
            }, replaceId);
        }
    });

    // 页面卸载提示：上传中切走会中断，提示用户确认
    window.addEventListener("beforeunload", (e) => {
        if (currentXhr && currentXhr.readyState < 4) {
            e.preventDefault();
            e.returnValue = "";
            currentXhr.abort();
        }
    });
})();
