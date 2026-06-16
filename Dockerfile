# Dockerfile - AI 智能文档问答系统
# 单 worker（Embedding/Rerank 模型常驻内存，多 worker 会重复占用约 3GB）

FROM python:3.12-slim AS base

# 系统依赖：build-essential 编译 C 扩展，libgomp1 OpenMP（PyTorch/FlagEmbedding 需要）
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# 安装 uv（用官方镜像，比 pip 快且锁文件一致）
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

# 先复制依赖清单（利用 Docker 层缓存，依赖不变时不重装）
COPY pyproject.toml uv.lock ./

# 用 uv sync 安装依赖（--frozen 严格按锁文件，--no-dev 不装 dev 依赖）
RUN uv sync --frozen --no-dev --no-install-project

# 复制项目代码
COPY app/ ./app/
COPY static/ ./static/
COPY tests/ ./tests/

# 创建数据和模型目录（运行时挂载卷覆盖）
RUN mkdir -p /app/data /app/models

# 环境变量默认值（docker-compose 的 env_file 会覆盖）
ENV UV_PROJECT_ENVIRONMENT=/app/.venv \
    PATH="/app/.venv/bin:$PATH"

EXPOSE 8000

# 健康检查（模型加载约 30s，start-period 给 60s 余量）
HEALTHCHECK --interval=30s --timeout=5s --start-period=60s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health', timeout=3)" || exit 1

# gunicorn 单 worker + uvicorn worker class（生产级进程管理）
# 单 worker 因为模型常驻内存；--timeout 120 覆盖大文档解析 + LLM 流式
CMD ["gunicorn", "app.main:app", "-w", "1", "-k", "uvicorn.workers.UvicornWorker", "-b", "0.0.0.0:8000", "--timeout", "120"]
