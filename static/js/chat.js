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

    // ---------- 会话管理 ----------
    let currentConvId = null; // 当前会话 id
    const convList = document.getElementById("convList");
    const newConvBtn = document.getElementById("newConvBtn");
    const chatSidebar = document.getElementById("chatSidebar");

    // Toast
    const toastEl = document.getElementById("toast");
    const toast = new bootstrap.Toast(toastEl, { delay: 2500 });
    function showToast(msg, type = "primary") {
        toastEl.className = `toast align-items-center text-bg-${type} border-0`;
        document.getElementById("toastBody").textContent = msg;
        toast.show();
    }

    // 移动端侧边栏切换
    document.getElementById("sidebarToggle").addEventListener("click", () => {
        chatSidebar.classList.toggle("open");
    });

    /** 渲染会话列表 */
    function renderConvList(convs) {
        if (!convs.length) {
            convList.innerHTML = `<div class="text-center text-muted small py-4">暂无会话，点击"新建"开始</div>`;
            return;
        }
        convList.innerHTML = convs
            .map((c) => {
                const active = c.id === currentConvId ? "active" : "";
                return `<div class="conv-item ${active}" data-id="${c.id}">
                    <span class="conv-title" title="${escapeHtml(c.title)}">${escapeHtml(c.title)}</span>
                    <button class="conv-del" title="删除"><i class="bi bi-trash"></i></button>
                </div>`;
            })
            .join("");
    }

    /** 加载会话列表 */
    async function loadConversations() {
        try {
            const data = await window.API.fetchJSON(ENDPOINTS.chat.conversations);
            renderConvList(data.conversations || []);
            return data.conversations || [];
        } catch (err) {
            convList.innerHTML = `<div class="text-danger small p-2">${escapeHtml(err.message)}</div>`;
            return [];
        }
    }

    /** 新建会话 */
    async function createConversation() {
        try {
            const data = await window.API.fetchJSON(ENDPOINTS.chat.conversations, { method: "POST" });
            currentConvId = data.id;
            await loadConversations();
            // 高亮新会话 + 清空 chatBox
            renderConvList((await loadConversations()) || []);
            highlightConv(currentConvId);
            clearChatBox();
            closeSidebarMobile();
        } catch (err) {
            showToast(err.message, "danger");
        }
    }

    function highlightConv(id) {
        convList.querySelectorAll(".conv-item").forEach((el) => {
            el.classList.toggle("active", Number(el.dataset.id) === id);
        });
    }

    function closeSidebarMobile() {
        chatSidebar.classList.remove("open");
    }

    function clearChatBox() {
        chatBox.innerHTML = `<div class="text-center text-muted py-5">
            <i class="bi bi-robot fs-1 d-block mb-3"></i>
            <h5 class="fw-normal">开始新的对话</h5>
            <p class="small">在下方输入你的问题</p>
        </div>`;
    }

    /** 切换会话：加载历史消息 */
    async function switchConversation(id) {
        if (streaming) {
            showToast("请等待当前回答完成", "warning");
            return;
        }
        currentConvId = id;
        highlightConv(id);
        closeSidebarMobile();
        chatBox.innerHTML = `<div class="text-center text-muted py-5"><div class="spinner-border spinner-border-sm"></div> 加载历史...</div>`;
        try {
            const data = await window.API.fetchJSON(
                `${ENDPOINTS.chat.history}?conversation_id=${id}&limit=50`
            );
            renderHistoryMessages(data.messages || []);
        } catch (err) {
            chatBox.innerHTML = `<div class="text-center text-danger py-5">${escapeHtml(err.message)}</div>`;
        }
    }

    /** 渲染历史消息（区分 user/assistant，复用 markdown + reasoning + sources 渲染）*/
    function renderHistoryMessages(messages) {
        if (!messages.length) {
            clearChatBox();
            return;
        }
        chatBox.innerHTML = "";
        // messages 是倒序的（最新在前），反转为正序渲染
        messages.slice().reverse().forEach((m) => {
            if (m.role === "user") {
                appendUserMsg(m.content);
            } else {
                const { contentEl, reasoningBox, reasoningEl, sourcesArea, reasoningHint } = createAssistantMsg();
                // 渲染完整答案
                contentEl.innerHTML = renderMarkdown(m.content);
                contentEl.querySelectorAll("pre code").forEach((b) => {
                    if (window.hljs) try { window.hljs.highlightElement(b); } catch {}
                });
                // 推理过程（如有）：历史消息默认折叠，避免占用大量纵向空间
                if (m.reasoning) {
                    reasoningEl.innerHTML = `<pre class="reasoning-pre">${escapeHtml(m.reasoning)}</pre>`;
                    reasoningBox.style.display = "block";
                    reasoningBox.open = false;
                    if (reasoningHint) reasoningHint.textContent = "点击展开/收起";
                } else {
                    // 无推理内容则隐藏面板
                    reasoningBox.style.display = "none";
                }
                // 来源（如有）
                if (m.sources && m.sources.length) {
                    renderSources(sourcesArea, m.sources);
                }
            }
        });
        chatBox.scrollTop = chatBox.scrollHeight;
    }

    /** 删除会话（事件委托）*/
    convList.addEventListener("click", async (e) => {
        const delBtn = e.target.closest(".conv-del");
        if (delBtn) {
            e.stopPropagation();
            const item = delBtn.closest(".conv-item");
            const id = Number(item.dataset.id);
            if (!confirm("确定删除该会话？所有消息将清除。")) return;
            try {
                await window.API.fetchJSON(ENDPOINTS.chat.deleteConversation(id), { method: "DELETE" });
                showToast("删除成功", "success");
                // 若删的是当前会话，切换到第一个或新建
                if (id === currentConvId) {
                    const convs = await loadConversations();
                    if (convs.length) {
                        await switchConversation(convs[0].id);
                    } else {
                        currentConvId = null;
                        clearChatBox();
                    }
                } else {
                    await loadConversations();
                    highlightConv(currentConvId);
                }
            } catch (err) {
                showToast(err.message, "danger");
            }
            return;
        }
        // 点击会话项 → 切换
        const item = e.target.closest(".conv-item");
        if (item) {
            const id = Number(item.dataset.id);
            if (id !== currentConvId) {
                await switchConversation(id);
            }
        }
    });

    newConvBtn.addEventListener("click", createConversation);

    // 页面初始化：加载会话列表 + 自动选中第一个
    (async function init() {
        const convs = await loadConversations();
        if (convs.length) {
            await switchConversation(convs[0].id);
        }
    })();

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
     * - sourcesArea 在气泡内顶部（参考来源 chip 标注）
     * - reasoningBox 推理折叠面板，初始隐藏
     */
    function createAssistantMsg() {
        const row = document.createElement("div");
        row.className = "msg-row assistant-row";
        row.innerHTML = `
            <div class="bubble assistant-bubble">
                <div class="sources-area mb-2"></div>
                <details class="reasoning-panel mb-2" style="display:none" open>
                    <summary class="reasoning-summary">
                        <i class="bi bi-lightbulb me-1"></i>推理过程
                        <span class="reasoning-hint small text-muted ms-1">思考中...</span>
                    </summary>
                    <div class="reasoning-content mt-1"></div>
                </details>
                <div class="assistant-content"><span class="typing-cursor"></span></div>
            </div>
        `;
        chatBox.appendChild(row);
        const contentEl = row.querySelector(".assistant-content");
        const reasoningBox = row.querySelector(".reasoning-panel");
        const reasoningEl = row.querySelector(".reasoning-content");
        const sourcesArea = row.querySelector(".sources-area");
        const reasoningHint = row.querySelector(".reasoning-hint");
        return { row, contentEl, reasoningBox, reasoningEl, sourcesArea, reasoningHint };
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

    // ---------- 来源标注（气泡内顶部 chip 标签，纯展示）----------
    function renderSources(areaEl, sources) {
        if (!sources || !sources.length) return;
        // 提取不重复的文件名（多个 chunk 可能来自同一文档）
        const names = [...new Set(sources.map((s) => s.filename).filter(Boolean))];
        if (!names.length) return;
        const chips = names
            .map((n) => `<span class="source-chip"><i class="bi bi-file-earmark-text me-1"></i>${escapeHtml(n)}</span>`)
            .join("");
        areaEl.innerHTML = `<span class="sources-label me-1"><i class="bi bi-quote"></i> 参考</span>${chips}`;
    }

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

        const { contentEl, reasoningBox, reasoningEl, sourcesArea, reasoningHint } = createAssistantMsg();
        let fullAnswer = "";
        let fullReasoning = "";
        let reasoningDone = false; // 推理完成（收到首个 token）后自动收起面板

        function updateContent() {
            contentEl.innerHTML = renderMarkdown(fullAnswer) + `<span class="typing-cursor"></span>`;
            renderMarkdownHighlight(contentEl);
            scrollToBottom();
        }

        function updateReasoning() {
            // 推理内容用纯文本 + 换行保留（不渲染 markdown，避免与正文混淆）
            reasoningEl.innerHTML = `<pre class="reasoning-pre">${escapeHtml(fullReasoning)}</pre>`;
            reasoningBox.style.display = "block";
            // 流式中保持展开，让用户看到思考过程在增长（避免误以为"卡住无输出"）
            if (!reasoningDone) reasoningBox.open = true;
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
                // 首个正式 token 到达 → 推理阶段结束，收起推理面板，让正文成为焦点
                if (!reasoningDone && fullAnswer === "") {
                    reasoningDone = true;
                    reasoningBox.open = false;
                    if (reasoningHint) reasoningHint.textContent = "点击展开/收起";
                }
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
                // 流结束：若从未产生正式 token（纯思考/异常），保持面板展开让用户看到思考；
                // 已生成答案则收起，并把提示文案改回静态
                if (fullAnswer && !reasoningDone) {
                    reasoningBox.open = false;
                    if (reasoningHint) reasoningHint.textContent = "点击展开/收起";
                } else if (reasoningHint && fullReasoning) {
                    reasoningHint.textContent = "点击展开/收起";
                }
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
            // AbortController：60s 无数据则中止（防后端卡死），每收到帧重置
            const controller = new AbortController();
            let abortTimer;
            const resetAbortTimer = () => {
                clearTimeout(abortTimer);
                abortTimer = setTimeout(() => controller.abort(), 60000);
            };
            window._chatAbortController = controller; // beforeunload 时可引用

            const resp = await fetch(ENDPOINTS.chat.ask, {
                method: "POST",
                headers: {
                    "Content-Type": "application/json",
                    ...(token ? { Authorization: `Bearer ${token}` } : {}),
                },
                body: JSON.stringify({ question, thinking, conversation_id: currentConvId }),
                signal: controller.signal,
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
            resetAbortTimer(); // 开始计时
            while (true) {
                const { done, value } = await reader.read();
                if (done) break;
                resetAbortTimer(); // 每收到数据重置（有流动不超时）
                parser.feed(decoder.decode(value, { stream: true }));
            }
            clearTimeout(abortTimer);
            parser.flush();

            // 流结束后移除打字光标
            const cursor = contentEl.querySelector(".typing-cursor");
            if (cursor) cursor.remove();
        } catch (err) {
            clearTimeout(abortTimer);
            if (err.name === "AbortError") {
                contentEl.innerHTML += `<div class="text-muted small mt-2"><i class="bi bi-clock-history me-1"></i>已中断（超时或离开页面）</div>`;
                statusHint.textContent = "已中断";
            } else {
                contentEl.innerHTML = `<div class="text-danger"><i class="bi bi-exclamation-triangle me-1"></i>${escapeHtml(err.message)}</div>`;
                statusHint.textContent = "出错";
            }
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

    // 页面卸载提示：生成中切走会中断，提示用户确认
    window.addEventListener("beforeunload", (e) => {
        if (streaming) {
            e.preventDefault();
            e.returnValue = "";
            window._chatAbortController?.abort();
        }
    });
})();
