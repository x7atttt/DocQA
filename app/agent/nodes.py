from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_deepseek import ChatDeepSeek

from app.agent.state import AgentState
from app.config import get_settings
from app.services.document_service import get_user_collection
from app.services.embedding_service import encode_single
from app.services.rerank_service import rerank

settings = get_settings()

# 最近 N 轮历史作为上下文传给 LLM（每轮 = user + assistant 两条消息）
MAX_HISTORY_ROUNDS = 5

_llm: ChatDeepSeek | None = None          # intent_router 用（非流式，无 thinking）
_llm_stream: ChatDeepSeek | None = None   # 流式，无 thinking
_llm_stream_thinking: ChatDeepSeek | None = None  # 流式，开启 thinking


def _make_llm(streaming: bool, thinking: bool) -> ChatDeepSeek:
    """构造 ChatDeepSeek。thinking=True 时透传 DeepSeek 的 thinking 开关。

    用 langchain-deepseek 而非 langchain-openai：前者原生解析 reasoning_content
    到 chunk.additional_kwargs['reasoning_content']，支持 thinking 流式推理捕获。
    """
    kwargs = dict(
        api_key=settings.llm_api_key,
        model=settings.llm_model,
        base_url=settings.llm_base_url,  # 保持可配置（兼容自定义 endpoint）
    )
    if streaming:
        kwargs["streaming"] = True
        kwargs["max_tokens"] = 1024
    else:
        kwargs["max_tokens"] = 512
    if thinking:
        # DeepSeek 思考模式：通过 extra_body 透传
        kwargs["extra_body"] = {"thinking": {"type": "enabled"}}
    return ChatDeepSeek(**kwargs)


def get_llm(streaming: bool = False, thinking: bool = False) -> ChatDeepSeek:
    """获取 LLM 实例（带缓存）。

    - intent_router 用 streaming=False, thinking=False
    - 答案生成用 streaming=True，thinking 由用户请求决定
    """
    global _llm, _llm_stream, _llm_stream_thinking
    if not streaming and not thinking:
        if _llm is None:
            _llm = _make_llm(streaming=False, thinking=False)
        return _llm
    if streaming and not thinking:
        if _llm_stream is None:
            _llm_stream = _make_llm(streaming=True, thinking=False)
        return _llm_stream
    if streaming and thinking:
        if _llm_stream_thinking is None:
            _llm_stream_thinking = _make_llm(streaming=True, thinking=True)
        return _llm_stream_thinking
    return _make_llm(streaming=False, thinking=True)


async def intent_router(state: AgentState) -> AgentState:
    question = state["question"]
    user_id = state["user_id"]

    collection = get_user_collection(user_id)
    try:
        doc_count = collection.count()
    except Exception:
        doc_count = 0

    if doc_count == 0:
        state["should_retrieve"] = False
        return state

    try:
        llm = get_llm()
        messages = [
            SystemMessage(
                content=(
                    "判断用户问题是否需要从用户上传的私有文档中检索信息。只输出 yes 或 no。\n"
                    "输出 yes 的情况：问题询问文档内容、要求总结/查找/对比具体信息、提到'文档/资料/文件/上面提到'等；"
                    "包含指代词（它/这个/那个）且上下文可能指向文档内容时也输出 yes。\n"
                    "输出 no 的情况：纯闲聊、写代码、通用百科知识、创作类请求。\n"
                    "拿不准时倾向输出 yes（宁可多检索）。"
                )
            ),
            HumanMessage(content=f"问题：{question}"),
        ]
        resp = await llm.ainvoke(messages)
        state["should_retrieve"] = "yes" in resp.content.strip().lower()
    except Exception:
        state["should_retrieve"] = True
    return state


async def retrieve_documents(state: AgentState) -> AgentState:
    question = state["question"]
    user_id = state["user_id"]

    collection = get_user_collection(user_id)
    query_vec = await encode_single(question)

    try:
        results = collection.query(query_embeddings=[query_vec], n_results=settings.retrieve_top_k)
    except Exception:
        state["retrieved_docs"] = []
        state["sources"] = []
        return state

    candidates = results.get("documents", [[]])[0]
    metadatas = results.get("metadatas", [[]])[0]
    if not candidates:
        state["retrieved_docs"] = []
        state["sources"] = []
        return state

    top_pairs = await rerank(question, candidates, top_k=settings.rerank_top_k)

    retrieved_docs: list[str] = []
    sources = []
    for orig_idx, score in top_pairs:
        retrieved_docs.append(candidates[orig_idx])
        meta = metadatas[orig_idx]
        sources.append(
            {
                "document_id": meta.get("document_id"),
                "filename": meta.get("filename", ""),
                "chunk_index": meta.get("chunk_index", 0),
                "content": candidates[orig_idx][:200],
                "score": round(float(score), 4),
            }
        )

    state["retrieved_docs"] = retrieved_docs
    state["sources"] = sources
    return state


def _history_to_messages(history: list[dict]) -> list:
    """把 state['history'] 转成 LangChain 消息列表（正序：最旧在前）。"""
    msgs = []
    for item in history[-MAX_HISTORY_ROUNDS * 2 :]:  # 最多取最近 N 轮
        role = item.get("role")
        content = item.get("content", "")
        if role == "user":
            msgs.append(HumanMessage(content=content))
        elif role == "assistant":
            msgs.append(AIMessage(content=content))
    return msgs


def _build_rag_prompt(question: str, context_docs: list[str], history: list[dict] | None = None) -> list:
    context = "\n\n---\n\n".join(context_docs) if context_docs else "(无相关文档)"
    system = (
        "你是文档问答助手。基于以下文档内容回答用户问题。"
        "要求：1) 答案必须仅基于文档内容；2) 若文档无法回答请直接说明'根据当前文档无法回答'，不要编造；"
        "3) 简洁专业，中文回答。"
    )
    user = f"文档内容：\n{context}\n\n用户问题：{question}"
    # 消息顺序：system → 历史 → 当前 human（让模型理解指代与上下文）
    messages = [SystemMessage(content=system)]
    if history:
        messages.extend(_history_to_messages(history))
    messages.append(HumanMessage(content=user))
    return messages


async def generate_answer(state: AgentState) -> AgentState:
    question = state["question"]
    docs = state.get("retrieved_docs", [])
    history = state.get("history", [])
    thinking = bool(state.get("thinking", False))

    if not docs:
        state["answer"] = "根据当前文档无法回答该问题。"
        state["answer_tokens"] = [state["answer"]]
        return state

    # ChatDeepSeek 原生支持 reasoning_content，thinking 模式下流式 reasoning 会
    # 出现在 chunk.additional_kwargs['reasoning_content']，由 chat_service 捕获
    llm = get_llm(streaming=True, thinking=thinking)
    messages = _build_rag_prompt(question, docs, history)

    tokens: list[str] = []
    async for chunk in llm.astream(messages):
        content = chunk.content
        if isinstance(content, str) and content:
            tokens.append(content)

    state["answer_tokens"] = tokens
    state["answer"] = "".join(tokens)
    return state


async def general_answer(state: AgentState) -> AgentState:
    question = state["question"]
    history = state.get("history", [])
    thinking = bool(state.get("thinking", False))

    llm = get_llm(streaming=True, thinking=thinking)
    messages = _history_to_messages(history)
    messages.append(HumanMessage(content=question))

    tokens: list[str] = []
    async for chunk in llm.astream(messages):
        content = chunk.content
        if isinstance(content, str) and content:
            tokens.append(content)

    state["answer_tokens"] = tokens
    state["answer"] = "".join(tokens)
    return state
