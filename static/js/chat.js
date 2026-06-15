/**
 * 对话页：SSE 流式问答 + 来源引用渲染 + 推理过程折叠
 *
 * 后端 SSE 协议（标准 event:/data: 帧）：
 *   event: sources       data: [SourceItem]        JSON，检索到的来源（token 之前最多发一次）
 *   event: reasoning     data: <原始字符串>          裸字符串，推理过程增量（DeepSeek reasoner 等）
 *   event: token         data: <原始字符串>          裸字符串，LLM 正式答案增量
 *   event: answer_final  data: {"answer":...[, "reasoning":...]}  完整答案/推理
 *   event: done          data: {"status":"ok"[,"cache":"hit"|"wait"]}
 *   event: error         data: {"message":"..."}
 */
(function () {
    "use strict";
    if (!window.API.requireAuth()) return;

    const { Token, ENDPOINTS, renderMarkdown, escapeHtml } = window.API;

    // 用户名
    const user = Token.getUser();
    if (user) document.getElementById("navUser").textContent = `👤 ${user.username}`;

    document.getElementById("logoutBtn").addEventListener("click", () => {
        Token.clear();
        location.href = "/login.html";
    });

    const chatBox = document.getElementById("chatBox");
    const askForm = document.getElementById("askForm");
    const questionInput = document.getElementById("questionInput");
    const sendBtn = document.getElementById("sendBtn");
    const statusHint = document.getElementById("statusHint");
    const thinkingToggle = document.getElementById("thinkingToggle");

    // 深度思考开关：用 localStorage 记住用户偏好
    const THINKING_KEY = "docqa_thinking";
    thinkingToggle.checked = localStorage.getItem(THINKING_KEY) === "1";
    thinkingToggle.addEventListener("change", () => {
        localStorage.setItem(THINKING_KEY, thinkingToggle.checked ? "1" : "0");
    });

    let streaming = false; // 是否正在接收流（防止并发）

    // ---------- DOM 渲染辅助 ----------
    function appendUserMsg(text) {
        const el = document.createElement("div");
        el.className = "msg-row user-row";
        el.innerHTML = `<div class="bubble user-bubble">${escapeHtml(text)}</div>`;
        chatBox.appendChild(el);
        scrollToBottom();
    }

    /**
     * 创建一条助手消息，返回：
     *   { row, contentEl, reasoningBox, reasoningEl, sourcesArea }
     * reasoningBox: 推理折叠面板容器（<details>），初始隐藏；收到 reasoning 事件才显示
     */
    function createAssistantMsg() {
        const row = document.createElement("div");
        row.className = "msg-row assistant-row";
        row.innerHTML = `
            <div class="bubble assistant-bubble">
                <details class="reasoning-panel mb-2" style="display:none">
                    <summary class="reasoning-summary">
                        <i class="bi bi-lightbulb me-1"></i>推理过程
                        <span class="reasoning-hint small text-muted ms-1">点击展开/收起</span>
                    </summary>
                    <div class="reasoning-content mt-1"></div>
                </details>
                <div class="assistant-content"><span class="typing-cursor"></span></div>
            </div>
            <div class="sources-area mt-2"></div>
        `;
        chatBox.appendChild(row);
        const contentEl = row.querySelector(".assistant-content");
        const reasoningBox = row.querySelector(".reasoning-panel");
        const reasoningEl = row.querySelector(".reasoning-content");
        const sourcesArea = row.querySelector(".sources-area");
        return { row, contentEl, reasoningBox, reasoningEl, sourcesArea };
    }

    function renderMarkdownHighlight(html) {
        // 代码高亮
        html.querySelectorAll("pre code").forEach((b) => {
            if (window.hljs) try { window.hljs.highlightElement(b); } catch {}
        });
    }

    function scrollToBottom() {
        chatBox.scrollTop = chatBox.scrollHeight;
    }

    // ---------- 来源卡片（点击展开完整原文）----------
    function renderSources(areaEl, sources) {
        if (!sources || !sources.length) return;
        const cards = sources
            .map((s, i) => {
                const name = escapeHtml(s.filename || "来源");
                const score = typeof s.score === "number" ? (s.score * 100).toFixed(0) + "%" : "";
                const fullContent = escapeHtml(s.content || "");
                const snippet =
                    escapeHtml((s.content || "").slice(0, 120)) + (s.content && s.content.length > 120 ? "…" : "");
                return `<div class="source-card" data-full="${fullContent}">
                    <div class="d-flex justify-content-between">
                        <span class="fw-semibold"><i class="bi bi-link-45deg me-1"></i>${i + 1}. ${name}</span>
                        <span>
                            ${score ? `<span class="badge bg-success-subtle text-success">${score}</span>` : ""}
                            <button class="btn btn-sm btn-link p-0 ms-1 src-expand"><i class="bi bi-arrows-expand"></i></button>
                        </span>
                    </div>
                    <div class="source-snippet small text-muted mt-1">${snippet}</div>
                </div>`;
            })
            .join("");
        areaEl.innerHTML = `<div class="sources-label small text-muted mb-1"><i class="bi bi-quote me-1"></i>参考来源（模型基于以下文档片段回答）</div>${cards}`;
    }

    // 点击来源卡片的展开按钮 → 切换显示完整原文
    document.addEventListener("click", (e) => {
        const btn = e.target.closest(".src-expand");
        if (!btn) return;
        const card = btn.closest(".source-card");
        const snippetEl = card.querySelector(".source-snippet");
        const full = card.dataset.full;
        if (card.dataset.expanded === "1") {
            // 收起：恢复摘要
            snippetEl.textContent = (card.dataset.origSnippet || "") + ((card.dataset.origFull || "").length > 120 ? "…" : "");
            card.dataset.expanded = "0";
            btn.innerHTML = `<i class="bi bi-arrows-expand"></i>`;
        } else {
            if (!card.dataset.origSnippet) {
                card.dataset.origSnippet = snippetEl.textContent.replace(/…$/, "");
                card.dataset.origFull = full;
            }
            snippetEl.textContent = full;
            card.dataset.expanded = "1";
            btn.innerHTML = `<i class="bi bi-arrows-collapse"></i>`;
        }
    });

    // ---------- SSE 帧解析器（跨 chunk 缓冲）----------
    function createSSEParser(handlers) {
        let buffer = "";
        return {
            feed(chunk) {
                buffer += chunk;
                let idx;
                while ((idx = buffer.indexOf("\n\n")) !== -1) {
                    const frame = buffer.slice(0, idx);
                    buffer = buffer.slice(idx + 2);
                    parseFrame(frame, handlers);
                }
            },
            flush() {
                if (buffer.trim()) parseFrame(buffer, handlers);
                buffer = "";
            },
        };
    }

    function parseFrame(frame, handlers) {
        let event = "message";
        const dataLines = [];
        frame.split("\n").forEach((line) => {
            if (line.startsWith("event:")) {
                event = line.slice(6).trim();
            } else if (line.startsWith("data:")) {
                dataLines.push(line.slice(5).replace(/^ /, ""));
            }
        });
        if (!dataLines.length) return;
        const raw = dataLines.join("\n");
        const h = handlers[event] || handlers.message;
        if (h) h(raw);
    }

    // ---------- 发送问答 ----------
    async function ask(question) {
        if (streaming) return;
        streaming = true;
        setSending(true);
        const thinking = thinkingToggle.checked;
        statusHint.textContent = thinking ? "深度思考中..." : "正在检索文档...";

        appendUserMsg(question);

        const { contentEl, reasoningBox, reasoningEl, sourcesArea } = createAssistantMsg();
        let fullAnswer = "";
        let fullReasoning = "";

        function updateContent() {
            contentEl.innerHTML = renderMarkdown(fullAnswer) + `<span class="typing-cursor"></span>`;
            renderMarkdownHighlight(contentEl);
            scrollToBottom();
        }

        function updateReasoning() {
            // 推理内容用纯文本 + 换行保留（不渲染 markdown，避免与正文混淆）
            reasoningEl.innerHTML = `<pre class="reasoning-pre">${escapeHtml(fullReasoning)}</pre>`;
            reasoningBox.style.display = "block";
            scrollToBottom();
        }

        const parser = createSSEParser({
            sources: (raw) => {
                try {
                    const arr = JSON.parse(raw);
                    renderSources(sourcesArea, arr);
                } catch {}
                statusHint.textContent = "正在生成回答...";
            },
            reasoning: (raw) => {
                fullReasoning += raw;
                updateReasoning();
                if (statusHint.textContent === "正在检索文档...") statusHint.textContent = "正在推理...";
            },
            token: (raw) => {
                fullAnswer += raw;
                updateContent();
                if (statusHint.textContent === "正在检索文档..." || statusHint.textContent === "正在推理...") {
                    statusHint.textContent = "正在生成回答...";
                }
            },
            answer_final: (raw) => {
                try {
                    const obj = JSON.parse(raw);
                    if (obj && typeof obj.answer === "string") fullAnswer = obj.answer;
                    if (obj && typeof obj.reasoning === "string" && obj.reasoning) {
                        fullReasoning = obj.reasoning;
                        updateReasoning();
                    }
                } catch {
                    if (raw) fullAnswer = raw;
                }
                updateContent();
            },
            done: (raw) => {
                let cacheTag = "";
                try {
                    const obj = JSON.parse(raw);
                    if (obj.cache === "hit") cacheTag = " · 缓存命中";
                    else if (obj.cache === "wait") cacheTag = " · 缓存等待命中";
                } catch {}
                statusHint.textContent = `完成${cacheTag}`;
            },
            error: (raw) => {
                let msg = "回答失败";
                try { msg = JSON.parse(raw).message || msg; } catch { if (raw) msg = raw; }
                contentEl.innerHTML = `<div class="text-danger"><i class="bi bi-exclamation-triangle me-1"></i>${escapeHtml(msg)}</div>`;
                statusHint.textContent = "出错";
            },
        });

        try {
            const token = Token.get();
            const thinking = thinkingToggle.checked;
            const resp = await fetch(ENDPOINTS.chat.ask, {
                method: "POST",
                headers: {
                    "Content-Type": "application/json",
                    ...(token ? { Authorization: `Bearer ${token}` } : {}),
                },
                body: JSON.stringify({ question, thinking }),
            });

            if (resp.status === 401) {
                Token.clear();
                location.href = "/login.html";
                return;
            }
            if (!resp.ok && !resp.headers.get("content-type")?.includes("text/event-stream")) {
                const payload = await resp.json().catch(() => null);
                throw new Error(payload?.message || `请求失败 (HTTP ${resp.status})`);
            }

            const reader = resp.body.getReader();
            const decoder = new TextDecoder();
            while (true) {
                const { done, value } = await reader.read();
                if (done) break;
                parser.feed(decoder.decode(value, { stream: true }));
            }
            parser.flush();

            // 流结束后移除打字光标
            const cursor = contentEl.querySelector(".typing-cursor");
            if (cursor) cursor.remove();
        } catch (err) {
            contentEl.innerHTML = `<div class="text-danger"><i class="bi bi-exclamation-triangle me-1"></i>${escapeHtml(err.message)}</div>`;
            statusHint.textContent = "出错";
        } finally {
            streaming = false;
            setSending(false);
            if (statusHint.textContent === "正在检索文档..." || statusHint.textContent === "正在生成回答..." || statusHint.textContent === "正在推理...") {
                statusHint.textContent = "完成";
            }
        }
    }

    function setSending(sending) {
        sendBtn.disabled = sending;
        questionInput.disabled = sending;
        if (!sending) questionInput.focus();
    }

    // ---------- 事件绑定 ----------
    askForm.addEventListener("submit", (e) => {
        e.preventDefault();
        const q = questionInput.value.trim();
        if (!q || streaming) return;
        questionInput.value = "";
        autoResize();
        ask(q);
    });

    questionInput.addEventListener("keydown", (e) => {
        if (e.key === "Enter" && !e.shiftKey) {
            e.preventDefault();
            askForm.requestSubmit();
        }
    });

    function autoResize() {
        questionInput.style.height = "auto";
        questionInput.style.height = Math.min(questionInput.scrollHeight, 160) + "px";
    }
    questionInput.addEventListener("input", autoResize);

    document.addEventListener("click", (e) => {
        if (e.target.classList.contains("example-q")) {
            e.preventDefault();
            questionInput.value = e.target.textContent;
            autoResize();
            questionInput.focus();
        }
    });
})();
