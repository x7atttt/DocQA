import logging
import os
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from slowapi import _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from slowapi.middleware import SlowAPIMiddleware

from app.api import auth, chat, documents
from app.core.cache import ping_redis
from app.core.database import init_db
from app.core.exceptions import register_exception_handlers
from app.core.rate_limit import limiter

logger = logging.getLogger("docqa")
logging.basicConfig(level=logging.INFO)


@asynccontextmanager
async def lifespan(app: FastAPI):
    os.makedirs("data", exist_ok=True)
    await init_db()
    try:
        await ping_redis()
        logger.info("Redis 连接正常")
    except Exception as e:
        logger.warning(f"Redis 连接失败：{e}。缓存与限流将不可用。")
    from app.agent.graph import build_graph

    app.state.graph = build_graph()
    yield


app = FastAPI(title="AI 智能文档问答系统", version="1.0.0", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)
app.add_middleware(SlowAPIMiddleware)

register_exception_handlers(app)

app.include_router(auth.router, prefix="/api/auth", tags=["认证"])
app.include_router(documents.router, prefix="/api/documents", tags=["文档"])
app.include_router(chat.router, prefix="/api/chat", tags=["对话"])


@app.get("/health")
async def health():
    return {"code": 0, "message": "ok", "data": {"status": "up"}}
