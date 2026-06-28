<div align="center">

# AI 智能文档问答系统（DocQA）

基于 **RAG（检索增强生成）** 的智能文档问答系统：上传文档 → 自动分块向量化 → 基于你的内容精准问答，答案带来源引用，响应实时流式输出。

FastAPI · LangGraph · ChromaDB · DeepSeek · Redis · BGE-M3

[功能特性](#-核心特性) · [快速开始](#-快速开始) · [API 文档](#-api-概览) · [项目结构](#-项目结构) · [部署](#-部署)

</div>

---

## 📋 目录

- [✨ 核心特性](#-核心特性)
- [🛠 技术栈](#-技术栈)
- [📁 项目结构](#-项目结构)
- [🧩 分块策略](#-分块策略可配置切换)
- [🧠 会话记忆](#-会话记忆)
- [🔍 检索与评测](#-检索与评测)
- [🚀 快速开始](#-快速开始)
- [📡 API 概览](#-api-概览)
- [🧪 测试](#-测试)
- [📦 部署](#-部署)
- [📝 License](#-license)

---

## ✨ 核心特性

| 特性 | 说明 |
|------|------|
| 📄 **多格式文档** | 支持 PDF / DOCX / Markdown。PDF 走 MinerU 高精度解析（OCR/表格/公式/图片提取，失败回退 pymupdf4llm），DOCX 走 MarkItDown，自动分块、Embedding 入库 |
| 📦 **批量上传** | 多文件选择 + 后端异步处理（BackgroundTasks）+ 状态轮询（pending→done），MinerU 并发限流（Semaphore）防云 API 限流 |
| 🧩 **多策略分块** | auto 按文件类型路由（md→markdown，pdf→html_preserve 保护表格，docx→recursive），HTMLSemanticPreservingSplitter 硬保 PDF 超大表格不被切 |
| 🔍 **RAG 检索增强** | BGE-M3 向量检索（dense+sparse RRF 混合召回）+ BGE-Reranker-v2-M3 交叉编码器精排 |
| 🤖 **LangGraph Agent** | 状态图编排"意图判断 → query 改写 → 检索 → 生成"链路，意图路由按需检索 |
| 💬 **多会话管理** | 侧边栏会话列表，新建/切换/删除，每用户上限 10 个，首问自动生成标题 |
| 💡 **深度思考模式** | DeepSeek thinking 开关，显式控制思考模式启停，推理过程可折叠查看 |
| ⚡ **SSE 流式输出** | Server-Sent Events 实时打字机效果，推理面板自动展开、答案逐字呈现 |
| 🧠 **智能降级回答** | 检索低相关/无命中时降级为「文档背景 + 常识」回答，泛化问题（如「怎么改进简历」）不再硬拒绝 |
| 🔁 **多轮上下文 + query 改写** | 保留会话最近若干轮历史（token 预算 + 轮数双约束截断，替代固定条数），检索前 LLM 改写指代词（"它""第三条"→ 完整问题），解决多轮指代检索失效 |
| 🧠 **长期记忆：会话摘要** | 长对话达阈值轮数后异步压缩老对话成摘要，注入 system prompt（"摘要 + 近期原文"混合上下文，Dify/FastGPT 同款），解决深挖文档时早期上下文丢失 |
| 🔄 **同名文档更新** | 上传同名文档时弹窗确认（replace_id 显式声明），避免误伤同名不同文档 |
| 🚀 **GPU 加速** | 自动检测 CUDA，BGE-M3/Reranker 在 GPU 上推理（encode 17 chunks：21.5s → 0.2s）|
| 🔐 **JWT 认证** | 注册/登录、数据按用户隔离 |
| 🚀 **Redis 缓存防护 + 限流** | 防穿透（空值缓存）/ 防击穿（SETNX 互斥锁）/ 防雪崩（TTL ±20% 抖动）+ slowapi 令牌桶限流（100/分钟/用户）|
| ⚡ **历史缓存** | 热点会话历史 Redis 缓存（按会话维度），写消息/删会话时失效，避免每次问答重查 DB |
| 🚫 **幂等上传** | sha256 文件哈希查重 + 数据库 `(user_id, file_hash)` 唯一约束兜底，防重复提交 |
| 📃 **游标分页** | 文档列表与对话历史均用 cursor 分页，无深分页性能问题 |

---

## 🛠 技术栈

| 层 | 技术 |
|------|------|
| **后端** | FastAPI · LangGraph · SQLAlchemy · Pydantic v2 |
| **RAG** | ChromaDB · FlagEmbedding（BGE-M3 稠密+稀疏向量 / BGE-Reranker-v2-M3 精排）|
| **LLM** | DeepSeek（OpenAI SDK 直连，兼容 OpenAI 格式，reasoning_content 自取）|
| **解析** | MinerU 云 API（PDF: OCR/表格/公式/图片，失败回退 pymupdf4llm）· MarkItDown（DOCX）|
| **分块** | langchain-text-splitters（MarkdownHeader / RecursiveCharacter）|
| **缓存/限流** | Redis · slowapi |
| **存储** | SQLite (aiosqlite) |
| **前端** | 原生 HTML + JS + Bootstrap 5 · marked.js · DOMPurify · highlight.js |
| **包管理** | uv · Python 3.12 |
| **部署** | Docker · gunicorn + uvicorn worker |

---

## 📁 项目结构

```
.
├── app/
│   ├── main.py              # FastAPI 入口 + 静态挂载
│   ├── config.py            # 配置（pydantic-settings）
│   ├── api/                 # 路由层：auth / documents / chat
│   ├── agent/               # LangGraph 状态图与节点编排
│   │   ├── graph.py         # 状态图定义（意图判断→改写→检索→生成）
│   │   ├── nodes.py         # 节点实现（query改写/混合检索/降级分流）
│   │   ├── memory.py        # 记忆工具（token 估算 + 预算截断 + 摘要注入）
│   │   └── state.py         # Agent 状态定义
│   ├── core/                # 安全/缓存/限流/响应/异常/数据库
│   ├── models/              # SQLAlchemy 模型（user/document/conversation）
│   ├── schemas/             # Pydantic 请求/响应模型
│   └── services/            # 业务层：文档/Embedding/Rerank/对话/分块/LLM
│       ├── chat_service.py  # SSE 流式核心（手动编排节点 + OpenAI SDK 流式）
│       ├── llm_provider.py  # LLM Provider 层（OpenAI SDK 直连，thinking 开关）
│       ├── summary_service.py  # 会话摘要（长对话压缩，异步触发 + SETNX 防重）
│       ├── ragas_embed_adapter.py  # BGE-M3→RAGAS Embeddings 适配器
│       ├── document_service.py  # 文档解析（MinerU/MarkItDown）+ 异步处理 + MinerU限流
│       ├── text_splitter.py # 多策略分块（含 html_preserve 表格保护）
│       ├── embedding_service.py  # BGE-M3（dense+sparse 双编码，device 自适应）
│       └── rerank_service.py
├── static/                  # 前端静态资源
│   ├── index.html           # 落地页
│   ├── login.html           # 登录/注册
│   ├── documents.html       # 文档管理
│   ├── chat.html            # 对话页（SSE 流式）
│   ├── css/style.css
│   └── js/                  # common / auth / documents / chat
├── tests/                   # pytest 测试（含综合篇.pdf 样本）
├── models/                  # 本地模型权重（bge-m3, bge-reranker-v2-m3）
├── data/                    # SQLite + ChromaDB 持久化（运行时生成）
├── Dockerfile
├── docker-compose.yml
├── pyproject.toml
└── .env                     # 环境变量（自行创建，参考 .env.example）
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

> 实现见 [`app/services/text_splitter.py`](app/services/text_splitter.py)，测试见 [`tests/test_chunking.py`](tests/test_chunking.py)（含 HTML 表格保护测试）。

---

## 🧠 会话记忆

多轮对话的上下文管理，分短期（窗口截断）和长期（会话摘要）两层：

| 层 | 机制 | 解决的问题 |
|------|------|---------|
| **短期：窗口截断** | 从最新历史向前累加，受 token 预算（`HISTORY_TOKEN_BUDGET`）+ 轮数（`MAX_HISTORY_ROUNDS`）双约束 | RAG 答案动辄 2-4k token，纯按条数截断会撑爆 context；按 token 控制总量更稳 |
| **长期：会话摘要** | 累计轮数达阈值（`SUMMARIZE_ROUND_THRESHOLD`）后异步压缩老对话成摘要，注入 system prompt | 深挖一篇文档时，早期问答超窗口后丢失，后续指代无法消解 |

**会话摘要流程**（Dify/FastGPT 同款"摘要 + 近期原文"方案）：

```
累计轮数 ≥ 阈值（默认12轮）
  ↓ BackgroundTasks 异步触发（不阻塞用户请求）
取【窗口外】的老对话（保留近期原文不压缩）
  ↓ LLM 压缩成要点摘要
写回 Conversation.summary，下次问答注入 system prompt
```

- **防重复**：摘要专用 SETNX 锁（按会话 id，60s），避免连续多轮触发重复生成
- **降级**：LLM 异常 / Redis 不可用都不阻断主流程，只记日志
- **历史缓存**：热点会话历史 Redis 缓存（按会话维度），写消息/删会话时失效

> 实现见 [`app/agent/memory.py`](app/agent/memory.py) + [`app/services/summary_service.py`](app/services/summary_service.py)，测试见 [`tests/test_memory.py`](tests/test_memory.py) / [`tests/test_summary.py`](tests/test_summary.py)。

---

## 🔍 检索与评测

### 检索架构（两阶段）

```
用户问题
  ↓ (query 改写：多轮指代消解)
BGE-M3 向量召回 Top-20（dense HNSW + sparse lexical RRF 混合融合）
  ↓
BGE-Reranker-v2-M3 交叉编码器精排 Top-3
  ↓
喂给 LLM 生成答案
```

- **召回**：BGE-M3 同时编码 dense 向量与 sparse lexical_weights，两路用 RRF（Reciprocal Rank Fusion）融合
- **精排**：BGE-Reranker cross-encoder 对候选独立打分重排，是检索质量的核心保障

### 评测体系（InduOCRBench）

基于 [qihoo360/InduOCRBench](https://huggingface.co/datasets/qihoo360/InduOCRBench) 中文企业文档评测集（12 行业 / 570 份 PDF / 2071 题）实测，详细记录见 [`rag指标记录.md`](rag指标记录.md)。

| 评测层 | 指标 | 结果 | 说明 |
|--------|------|:----:|------|
| 检索层（标准标注）| Hit@3 / MRR | **94% / 0.857** | 纯检索链路（rerank 精排使 MRR +12.9%）|
| 检索层（OCR→检索）| Hit@3 | **71%** | 含 MinerU 解析损耗（LLM-as-judge 判定）|

评测脚本在 `tests/eval/`，包含检索召回评测与 OCR→检索链路评测。

---

## 🚀 快速开始

### 1. 环境准备

- **Python 3.12+**
- **uv**（包管理）
- **Redis**（缓存与限流）
- 模型权重：bge-m3、bge-reranker-v2-m3 放到 `models/` 目录（可通过 [ModelScope](https://modelscope.cn) 下载）
- （可选）**NVIDIA GPU + CUDA**：自动启用 GPU 加速，无 GPU 自动回退 CPU

### 2. 安装依赖

```bash
uv sync
```

> **GPU 加速（可选）**：默认安装 CPU 版 torch。有 NVIDIA GPU 时，安装 CUDA 版可让 BGE-M3 encode 提速约 100 倍：
> ```bash
> uv add "torch==2.12.0+cu126" --index-strategy unsafe-best-match
> ```
> 代码已做 device 自适应（[`embedding_service.py`](app/services/embedding_service.py) / [`rerank_service.py`](app/services/rerank_service.py)），有无 GPU 都能跑。

### 3. 配置环境变量

创建 `.env` 文件（参考 [`.env.example`](.env.example)，关键字段）：

```dotenv
# LLM（DeepSeek，通过 OpenAI SDK 兼容接口直连）
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

# 会话记忆（历史窗口 / 摘要阈值，可不填用默认值）
# MAX_HISTORY_ROUNDS=5          # 生成答案时历史的最大轮数
# HISTORY_TOKEN_BUDGET=3500     # 历史的 token 预算上限（仅历史部分）
# SUMMARIZE_ROUND_THRESHOLD=12  # 累计轮数达此值触发会话摘要
# HISTORY_CACHE_TTL_SECONDS=7200 # 历史缓存 TTL（秒）
```

> 💡 **深度思考**：前端对话页有"深度思考"开关，开启后通过 DeepSeek `thinking` 模式输出推理过程（可折叠查看）。关闭时显式禁用思考模式（`deepseek-v4-flash` 默认开启思考，需显式 disabled），确保开关真正生效。这是请求级开关，无需配置。

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
```

首次启动会加载 Embedding/Rerank 模型（约 3GB 常驻内存），需耐心等待约 30 秒。启动完成后用浏览器访问：

- 🏠 **落地页**：http://127.0.0.1:8000/
- 📚 **API 文档**：http://127.0.0.1:8000/docs
- ❤️ **健康检查**：http://127.0.0.1:8000/health

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

主要覆盖：文档分块（15 用例）、Embedding 维度、认证、缓存、限流、会话记忆（token 截断/摘要注入/历史缓存）、query 改写、中英混排等。

---

## 📦 部署

### Docker 部署（推荐）

```bash
# 构建并启动（app + redis 双服务）
docker compose up -d --build

# 查看日志
docker compose logs -f app

# 健康检查
curl http://localhost:8000/health
```

`docker-compose.yml` 包含 `app` + `redis` 两个服务，数据卷挂载 `data/` 和 `models/`。

**生产部署要点**：
- 用 `gunicorn -k uvicorn.workers.UvicornWorker -w 1 --preload` 单 worker（Embedding 模型常驻内存，多 worker 会重复占用约 3GB）
- 反向代理（Nginx）需关闭 SSE 缓冲：`proxy_buffering off;`
- 模型权重挂载为卷，避免镜像过大
- 容器内默认 CPU 推理；需 GPU 时挂载 NVIDIA runtime 并安装 CUDA 版 torch

---

## 📝 License

[MIT](LICENSE)
