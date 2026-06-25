# AI 智能文档问答系统（DocQA）

基于 **RAG（检索增强生成）** 的智能文档问答系统：上传文档 → 自动分块向量化 → 基于你的内容精准问答，答案带来源引用，响应实时流式输出。

> 技术面试向个人项目。覆盖 后端架构 / RAG 全链路 / 流式输出 / 缓存限流 / 前端 等核心考点。

---

## ✨ 核心特性

| 特性 | 说明 |
|------|------|
| 📄 **多格式文档** | 支持 PDF / DOCX / Markdown。PDF 走 MinerU 高精度解析（OCR/表格/公式/图片提取，失败回退 pymupdf4llm），DOCX 走 MarkItDown，自动分块、Embedding 入库 |
| 🧩 **多策略分块** | auto 按文件类型路由（md→markdown，pdf/docx→recursive），可选 fixed/markdown/recursive |
| 🔍 **RAG 检索增强** | bge-m3 向量检索（稠密+稀疏+ColBERT 多视图）+ bge-reranker-v2-m3 重排序 |
| 🤖 **LangGraph Agent** | 状态图编排检索 → 重排 → 生成节点，意图路由按需检索 |
| 💬 **多会话管理** | 侧边栏会话列表，新建/切换/删除，每用户上限 10 个，首问自动生成标题 |
| 💡 **深度思考模式** | DeepSeek thinking 开关，推理过程可折叠查看（langchain-deepseek 原生 reasoning_content）|
| ⚡ **SSE 流式输出** | Server-Sent Events 实时打字机效果，答案逐字呈现 |
| 🔁 **多轮上下文** | 保留当前会话最近 5 轮历史，支持指代理解（"它""上面那个"）|
| 🚀 **多级缓存 + 限流** | Redis 缓存（按会话+thinking 模式分桶隔离，含空值防穿透）+ slowapi 令牌桶限流 |
| 🚫 **内容去重** | sha256 文件指纹，用户内去重，重复上传直接拒绝（省 embedding 算力）|
| 📊 **上传进度** | XMLHttpRequest 真实进度条 + 超时处理 + 重复文档友好提示 |
| 🔐 **JWT 认证** | 注册/登录、数据按用户隔离 |
| 📃 **游标分页** | 文档列表与对话历史均用 cursor 分页，无深分页性能问题 |

---

## 🛠 技术栈

```
后端      FastAPI · LangGraph · SQLAlchemy · Pydantic v2
RAG       ChromaDB · FlagEmbedding (bge-m3 / bge-reranker-v2-m3)
LLM       langchain-deepseek（原生 reasoning_content 流式捕获）
解析      MinerU 云 API（PDF: OCR/表格/公式/图片，失败回退 pymupdf4llm）· MarkItDown（DOCX）
分块      langchain-text-splitters（MarkdownHeader / RecursiveCharacter）
缓存/限流 Redis · slowapi
存储      SQLite (aiosqlite)
前端      原生 HTML + JS + Bootstrap 5 · marked.js · DOMPurify · highlight.js
包管理    uv · Python 3.12
```

---

## 🧩 分块策略（可配置切换）

通过 `.env` 的 `SPLIT_STRATEGY` 切换，无需改代码。默认 `auto`。

| 策略 | 原理 | 适用场景 |
|------|------|---------|
| **auto**（默认）| 按文件类型路由：`md→markdown`，`pdf/docx→recursive` | 大多数情况 |
| **markdown** | 两阶段：先按 `#/##/###` 标题切子节保结构，再对超长节递归切控长度 | 有清晰层级的文档（简历、报告、论文）|
| **recursive** | 递归尝试分隔符（段落→换行→句号→空格）切，兼顾语义边界与长度 | 通用场景、转换后标题不规整的 PDF/DOCX |
| **fixed** | 定长字符滑窗（带重叠），最简单但可能切断句子/表格 | 兜底/对照基准 |

**配置优先级**：`.env` 显式设置 > auto 按类型路由

**auto 路由设计依据**：
- 原生 `.md` 文件标题结构最完整，markdown 策略优势最大
- `pdf`/`docx` 经 MinerU/MarkItDown 转换后，标题层级按字号推断可能不规整，recursive 更稳健
- `fixed` 在任何场景都不如 recursive（recursive 最差情况退化为 fixed），故不作为自动选项
- 递归分隔符含 `</table>`，保护中小 HTML 表格不在分块时被切断（MinerU 表格输出为 HTML）

> 实现见 `app/services/text_splitter.py`，测试见 `tests/test_chunking.py`（含 HTML 表格保护测试）。

---

## 📁 项目结构

```
.
├── app/
│   ├── main.py              # FastAPI 入口 + 静态挂载
│   ├── config.py            # 配置（pydantic-settings）
│   ├── api/                 # 路由层：auth / documents / chat
│   ├── agent/               # LangGraph 状态图与节点编排
│   ├── core/                # 安全/缓存/限流/响应/异常/数据库
│   ├── models/              # SQLAlchemy 模型
│   ├── schemas/             # Pydantic 请求/响应模型
│   └── services/            # 业务层：文档/Embedding/Rerank/对话
├── static/                  # 前端静态资源
│   ├── index.html           # 落地页
│   ├── login.html           # 登录/注册
│   ├── documents.html       # 文档管理
│   ├── chat.html            # 对话页（SSE 流式）
│   ├── css/style.css
│   └── js/                  # common / auth / documents / chat
├── models/                  # 本地模型权重（bge-m3, bge-reranker-v2-m3）
├── data/                    # SQLite + ChromaDB 持久化（运行时生成）
├── tests/
├── pyproject.toml
└── .env                     # 环境变量（自行创建）
```

---

## 🚀 快速开始

### 1. 环境准备

- **Python 3.12+**
- **uv**（包管理）
- **Redis**（缓存与限流）
- 模型权重：bge-m3、bge-reranker-v2-m3 放到 `models/` 目录（可通过 ModelScope 下载）

### 2. 安装依赖

```bash
uv sync
```

### 3. 配置环境变量

创建 `.env` 文件（参考 `.env.example`，关键字段）：

```dotenv
# LLM（DeepSeek，通过 langchain-deepseek 接入）
LLM_BASE_URL=https://api.deepseek.com
LLM_API_KEY=sk-xxxxxxxx
LLM_MODEL=deepseek-v4-flash

# 数据库与缓存
DATABASE_URL=sqlite+aiosqlite:///./data/docqa.db
REDIS_URL=redis://localhost:6379/0

# JWT
JWT_SECRET_KEY=your-secret-key
JWT_ALGORITHM=HS256
JWT_EXPIRE_MINUTES=1440

# 模型路径（本地权重）
EMBEDDING_MODEL_PATH=./models/bge-m3
RERANK_MODEL_PATH=./models/bge-reranker-v2-m3

# 分块与检索参数
CHUNK_SIZE=500
CHUNK_OVERLAP=100
SPLIT_STRATEGY=auto
RETRIEVE_TOP_K=20
RERANK_TOP_K=3

# 会话管理
MAX_CONVERSATIONS=10
```

> 💡 **深度思考**：前端对话页有"深度思考"开关，开启后通过 DeepSeek `thinking` 模式输出推理过程（可折叠查看）。这是请求级开关，无需配置。

### 4. 启动 Redis

```bash
# 任选一种
redis-server                          # 本地直装
docker run -d -p 6379:6379 redis:7    # Docker
```

### 5. 启动服务

```bash
# 本地开发：监听本地回环地址
uv run uvicorn app.main:app --reload --host 127.0.0.1 --port 8000

# 需要局域网/容器外访问时，再监听所有网卡
# uv run uvicorn app.main:app --host 0.0.0.0 --port 8000
```

首次启动会加载 Embedding/Rerank 模型（约 3GB 常驻内存），需耐心等待约 30 秒。启动完成后用浏览器访问：

- 🏠 **落地页**：http://127.0.0.1:8000/
- 📚 **API 文档**：http://127.0.0.1:8000/docs
- ❤️ **健康检查**：http://127.0.0.1:8000/health

> ⚠️ **关于访问地址**：`0.0.0.0` 是**服务端监听地址**（让服务监听所有网卡），浏览器**无法直接访问**。客户端访问必须用 `127.0.0.1` 或 `localhost`（本机）或服务器的具体 IP（远程）。

### 6. 页面访问地址

| 页面 | 地址 | 说明 |
|------|------|------|
| 🏠 落地页 | http://127.0.0.1:8000/ | 项目介绍，未登录可看 |
| 🔐 登录/注册 | http://127.0.0.1:8000/static/login.html | 登录后存 JWT，自动跳对话 |
| 📚 文档管理 | http://127.0.0.1:8000/static/documents.html | 需登录：上传/删除/分页 |
| 💬 智能问答 | http://127.0.0.1:8000/static/chat.html | 需登录：SSE 流式问答 |
| 📖 API 文档 | http://127.0.0.1:8000/docs | Swagger UI |
| ❤️ 健康检查 | http://127.0.0.1:8000/health | 返回 `{"code":0,...}` |

> 已登录用户访问落地页会自动跳转到对话页。

---

## 📡 API 概览

所有 JSON 响应统一为 `{code, message, data}` 结构（`code=0` 表示成功）。除 `/auth/register`、`/auth/login` 外，所有接口需在请求头携带 `Authorization: Bearer <token>`。

| 方法 | 路径 | 说明 |
|------|------|------|
| POST | `/api/auth/register` | 注册 |
| POST | `/api/auth/login` | 登录，返回 JWT |
| POST | `/api/documents/upload` | 上传文档（multipart，字段名 `file`） |
| GET | `/api/documents/list` | 文档列表（游标分页 `cursor` / `limit`） |
| DELETE | `/api/documents/{id}` | 删除文档 |
| POST | `/api/chat/ask` | **SSE 流式问答**（可带 `conversation_id` 指定会话）|
| GET | `/api/chat/conversations` | 会话列表（按 updated_at 倒序，含 total） |
| POST | `/api/chat/conversations` | 新建空会话（达上限 10 个则拒绝）|
| DELETE | `/api/chat/conversations/{id}` | 删除会话（级联清除消息）|
| GET | `/api/chat/history` | 会话消息历史（必传 `conversation_id`，游标分页）|

### SSE 事件协议（`/api/chat/ask`）

```
event: sources        data: [<来源卡片>]        # 检索结果，token 之前最多发一次
event: reasoning      data: <裸字符串>          # 推理过程增量（thinking 模式开启时）
event: token          data: <裸字符串>          # LLM 正式答案增量，逐字推送
event: answer_final   data: {"answer":...,"reasoning":...}  # 完整答案与推理
event: done           data: {"status":"ok"}     # 结束（命中缓存时带 cache 字段）
event: error          data: {"message":"..."}   # 异常
```

---

## 🧪 测试

```bash
uv run pytest
```

主要覆盖：文档分块、Embedding 维度、中英混排等。

---

## 📦 部署

> ⚠️ Docker 部署待实现（`docker-compose.yml` / `Dockerfile` 目前为占位）。

当前推荐本地/裸机部署，启动命令同「快速开始」。生产部署注意：

- 用 `gunicorn -k uvicorn.workers.UvicornWorker -w 1` 单 worker（Embedding 模型常驻内存，多 worker 会重复占用）
- 反向代理（Nginx）需关闭 SSE 缓冲：`proxy_buffering off;`
- 模型权重挂载为卷，避免镜像过大

---

## 📌 开发阶段

| 阶段 | 内容 | 状态 |
|------|------|------|
| stage-1 | 基础骨架与 JWT 认证 | ✅ |
| stage-2 | 文档上传与向量入库 | ✅ |
| stage-3 | RAG Agent 与 SSE 流式对话 | ✅ |
| stage-4 | Redis 多级缓存 + slowapi 限流 | ✅ |
| stage-5 | 前端页面（落地/登录/文档/对话）| ✅ |
| stage-5 | langchain-deepseek + thinking 模式 + reasoning 折叠 | ✅ |
| stage-5 | 多轮上下文（5 轮）+ 来源标注 + 意图路由 | ✅ |
| stage-5 | 多策略分块（auto/markdown/recursive/fixed）| ✅ |
| stage-5 | 内容去重（sha256）+ 上传进度条 + 超时 | ✅ |
| stage-5 | 多会话管理（侧边栏/新建/切换/删除/上限10）| ✅ |
| stage-5 | Docker 容器化部署 | ⏳ |

---

## 📝 License

MIT
