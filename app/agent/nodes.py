from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_deepseek import ChatDeepSeek

from app.agent.state import AgentState
from app.config import get_settings
from app.services.document_service import get_user_collection
from app.services.embedding_service import encode_query_full, encode_single, sparse_score
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


async def rewrite_query(state: AgentState) -> AgentState:
    """多轮指代消解：结合历史把指代词改写成可独立检索的完整 query。

    场景：用户上一轮问"文档里有哪些规则"，这一轮问"第三条是什么"，
    直接用"第三条"检索必然失败。本节点把"第三条"结合历史改写成
    "文档里第3条规则的具体内容"。

    设计：
    - 仅当 history 非空才调 LLM 改写（首问无历史，省一次调用）
    - 改写结果存 rewritten_query，供 retrieve_documents 使用
    - 生成答案仍用原始 question（回答自然，用户问的是原话）
    - 容错：LLM 异常/输出异常时降级用原始 question，不阻断流程
    """
    question = state["question"]
    history = state.get("history", [])

    # 无历史 → 无指代可消解，直接用原问题
    if not history:
        state["rewritten_query"] = question
        return state

    try:
        llm = get_llm()
        # 把历史拼成对话格式（最多最近 3 轮，够消解指代又省 token）
        recent = history[-6:]  # 3 轮 = 6 条消息
        dialogue = "\n".join(
            f"{'用户' if h['role'] == 'user' else '助手'}: {h['content'][:200]}"
            for h in recent
            if h.get("content")
        )
        messages = [
            SystemMessage(
                content=(
                    "你是 query 改写助手。根据对话历史，把用户的最新问题改写成一个"
                    "可以独立检索的完整问题（消解指代词如 它/这个/第几条/上面提到）。\n"
                    "要求：\n"
                    "1) 只输出改写后的问题，不要任何解释；\n"
                    "2) 保持原意，不要添加无关内容；\n"
                    "3) 如果问题本身已完整无指代，原样输出。"
                )
            ),
            HumanMessage(content=f"对话历史：\n{dialogue}\n\n最新问题：{question}\n\n改写后："),
        ]
        resp = await llm.ainvoke(messages)
        # 去除首尾可能的多余引号/句号（LLM 有时会用引号包裹改写结果）
        # 含中文弯引号 “ ” ‘ ’
        rewritten = resp.content.strip().strip("\"'“”‘’。")
        state["rewritten_query"] = rewritten or question
    except Exception:
        # 改写失败不阻断检索，降级用原始问题
        state["rewritten_query"] = question
    return state


def _rrf_fuse(
    dense_rank: list[int], sparse_rank: list[int], k: int
) -> list[int]:
    """Reciprocal Rank Fusion：把 dense/sparse 两路的排名融合成统一候选序。

    dense_rank[i] 表示候选 i 在 dense 路的排名（0=最相关）；
    sparse_rank 同理。RRF 得分 = Σ 1/(k + rank)，得分越高越相关。
    返回按 RRF 得分降序的候选下标列表。
    """
    scores: dict[int, float] = {}
    for ranks in (dense_rank, sparse_rank):
        for rank, idx in enumerate(ranks):
            scores[idx] = scores.get(idx, 0.0) + 1.0 / (k + rank)
    return sorted(scores.keys(), key=lambda i: scores[i], reverse=True)


async def retrieve_documents(state: AgentState) -> AgentState:
    """Hybrid 检索：dense（Chroma HNSW）+ sparse（BGE-M3 lexical_weights）RRF 融合。

    流程（"dense 粗筛 → sparse 重排 → RRF 融合 → reranker 精排"）：
      1. BGE-M3 同时编码 query 的 dense 向量与 sparse lexical_weights
      2. dense 路：Chroma HNSW 召回 Top-N 候选（N=dense_recall_top_k=50）
      3. sparse 路：对这批候选重新算 lexical 匹配得分，给 sparse 排名
         （不遍历全库，只在 dense 候选集内重排，控制计算量）
      4. RRF 融合两路排名 → 取 Top-K（retrieve_top_k=20）
      5. BGE-Reranker 精排 → Top-3 喂生成

    相比纯 dense：sparse 路对精确术语/关键词/缩写的强匹配能补救 dense 的语义漂移，
    把"字面没对上但语义相关"和"字面正好对上"两类命中都纳入候选。
    """
    question = state["question"]
    # 检索用改写后的 query（指代消解），生成答案时仍用原始 question
    search_query = state.get("rewritten_query") or question
    user_id = state["user_id"]

    collection = get_user_collection(user_id)

    # 1. query 双编码：dense + sparse
    query_enc = await encode_query_full(search_query)
    query_vec = query_enc["dense"]
    query_sparse = query_enc["sparse"]

    try:
        # 2. dense 路：HNSW 召回候选（扩大到 dense_recall_top_k）
        results = collection.query(query_embeddings=[query_vec], n_results=settings.dense_recall_top_k)
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

    # 3. sparse 路：对候选集算 lexical 匹配得分
    sparse_scores = await sparse_score(query_sparse, candidates)
    # dense 路的排名就是 Chroma 返回顺序（HNSW 已按相似度排好）
    dense_rank = list(range(len(candidates)))
    # sparse 路按得分降序排
    sparse_rank = sorted(range(len(candidates)), key=lambda i: sparse_scores[i], reverse=True)

    # 4. RRF 融合 → 取 Top-K（retrieve_top_k）
    fused_order = _rrf_fuse(dense_rank, sparse_rank, settings.rrf_k)
    top_idxs = fused_order[: settings.retrieve_top_k]
    fused_candidates = [candidates[i] for i in top_idxs]
    fused_metas = [metadatas[i] for i in top_idxs]

    # 5. reranker 精排（用改写后的 query，与召回保持一致）
    top_pairs = await rerank(search_query, fused_candidates, top_k=settings.rerank_top_k)

    retrieved_docs: list[str] = []
    sources = []
    for orig_idx, score in top_pairs:
        retrieved_docs.append(fused_candidates[orig_idx])
        meta = fused_metas[orig_idx]
        sources.append(
            {
                "document_id": meta.get("document_id"),
                "filename": meta.get("filename", ""),
                "chunk_index": meta.get("chunk_index", 0),
                "content": fused_candidates[orig_idx][:200],
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


def _build_fallback_prompt(question: str, context_docs: list[str], history: list[dict] | None = None) -> list:
    """检索无直接命中时的降级 prompt：结合文档背景 + 常识给出有帮助的回答。

    与严格 RAG 的区别：允许模型在"文档未直接回答"时，基于文档提供的背景信息
    （如简历内容、项目描述）+ 通用知识给出建议/分析，而不是硬拒绝。
    典型场景：用户上传简历后问"怎么改进我的简历"——文档里有简历内容，
    但没有现成的改进建议，此时应结合简历实际情况给针对性建议。
    """
    context = "\n\n---\n\n".join(context_docs) if context_docs else "(用户未上传相关文档)"
    system = (
        "你是一个智能助手。用户上传了以下文档作为参考背景。\n"
        "请根据用户问题作答：\n"
        "1) 若问题能从文档直接找到答案，请基于文档内容回答；\n"
        "2) 若文档未直接涉及该问题（如询问建议、评价、改进方案），请结合文档中可见的实际情况"
        "（如简历内容、项目细节）与你的通用知识，给出具体、有针对性的回答；\n"
        "3) 回答开头用一句话说明依据来源（如『基于您上传的简历内容』或『文档未直接涉及，以下为通用建议』）；\n"
        "4) 简洁专业，中文回答。"
    )
    user = f"文档背景：\n{context}\n\n用户问题：{question}"
    messages = [SystemMessage(content=system)]
    if history:
        messages.extend(_history_to_messages(history))
    messages.append(HumanMessage(content=user))
    return messages


async def generate_answer(state: AgentState) -> AgentState:
    question = state["question"]
    docs = state.get("retrieved_docs", [])
    sources = state.get("sources", [])
    history = state.get("history", [])
    thinking = bool(state.get("thinking", False))

    # 根据检索结果相关度选择 prompt 策略：
    # - 高相关（top score ≥ 0.5）：严格 RAG，仅基于文档回答
    # - 低相关 / 无结果：降级 fallback，结合文档背景 + 常识给有帮助的回答
    #   典型场景：用户上传简历后问"怎么改进简历"——文档有简历内容但无现成建议，
    #   严格 RAG 会硬拒绝，fallback 让模型结合简历实际情况给针对性建议。
    top_score = sources[0].get("score", 0) if sources else 0
    if docs and top_score >= 0.5:
        messages = _build_rag_prompt(question, docs, history)
    else:
        messages = _build_fallback_prompt(question, docs, history)

    # ChatDeepSeek 原生支持 reasoning_content，thinking 模式下流式 reasoning 会
    # 出现在 chunk.additional_kwargs['reasoning_content']，由 chat_service 捕获
    llm = get_llm(streaming=True, thinking=thinking)

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
