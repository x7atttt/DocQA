import json
from collections.abc import AsyncIterator
from typing import Any

from app.agent.state import AgentState


def sse(event: str, data: Any) -> str:
    payload = json.dumps(data, ensure_ascii=False) if not isinstance(data, str) else data
    return f"event: {event}\ndata: {payload}\n\n"


# 手动编排节点执行（替代原 graph.astream_events 方案）：
# 原 LangChain ChatDeepSeek 的 on_chat_model_stream 事件在改用 OpenAI SDK 后不再触发，
# 故改为 stream_graph 直接调用节点函数，在生成阶段用 astream_chat 实时 yield token/reasoning。
# 这样流式推送完全可控，thinking 模式的 reasoning_content 能正确推送到前端。


async def stream_graph(
    graph,
    user_id: int,
    question: str,
    history: list[dict] | None = None,
    thinking: bool = False,
    summary: str | None = None,
) -> AsyncIterator[tuple[str, Any]]:
    """yield (event_name, payload). event ∈ reasoning/token/sources/done/error.

    手动编排（不走 graph.astream_events）：
    1. intent_router → 判断要不要检索
    2. retrieve_documents（含 rewrite_query）→ 推 sources
    3. generate_answer/general_answer → astream_chat 流式 yield token/reasoning

    放弃 LangGraph 的 astream_events：它依赖 LangChain 的 on_chat_model_stream 事件，
    改用 OpenAI SDK 直连后该事件不触发。手动编排让生成阶段的流式完全可控，
    thinking 模式的 reasoning_content 实时推送，让前端"深度思考"开关真正生效。
    """
    # 延迟导入避免循环依赖（nodes 依赖 llm_provider，不依赖本模块）
    from app.agent.nodes import (
        _build_fallback_prompt,
        _build_rag_prompt,
        _history_to_messages,
        general_answer,
        generate_answer,
        intent_router,
        retrieve_documents,
    )
    from app.services.llm_provider import astream_chat
    from langchain_core.messages import HumanMessage

    initial: AgentState = {
        "user_id": user_id,
        "question": question,
        "rewritten_query": "",
        "history": history or [],
        "summary": summary or "",
        "thinking": thinking,
        "should_retrieve": False,
        "retrieved_docs": [],
        "sources": [],
        "answer_tokens": [],
        "answer": "",
        "reasoning_tokens": [],
        "reasoning": "",
        "error": None,
        "meta": {},
    }

    answer_parts: list[str] = []
    reasoning_parts: list[str] = []

    try:
        state = initial
        # 1. 意图路由
        state = await intent_router(state)

        if state.get("should_retrieve"):
            # 2. 检索（含 query 改写，retrieve_documents 内部先跑 rewrite）
            # 注：retrieve_documents 依赖 rewritten_query，由 rewrite_query 节点填充。
            # 这里手动补一步 rewrite，再 retrieve（与 graph 编排一致）
            from app.agent.nodes import rewrite_query

            state = await rewrite_query(state)
            state = await retrieve_documents(state)
            # 推送来源
            if state.get("sources"):
                yield ("sources", state["sources"])
        else:
            state["retrieved_docs"] = []
            state["sources"] = []

        # 3. 构造生成消息
        question_text = state["question"]
        history_list = state.get("history", [])
        summary_text = state.get("summary") or None
        if state.get("should_retrieve"):
            docs = state.get("retrieved_docs", [])
            sources = state.get("sources", [])
            top_score = sources[0].get("score", 0) if sources else 0
            if docs and top_score >= 0.5:
                messages = _build_rag_prompt(question_text, docs, history_list, summary_text)
            else:
                messages = _build_fallback_prompt(question_text, docs, history_list, summary_text)
        else:
            # 纯对话（general_answer 路径）：摘要也注入，保持长对话连贯
            messages = _history_to_messages(history_list)
            from app.agent.nodes import _inject_summary

            messages = _inject_summary(messages, summary_text)
            messages.append(HumanMessage(content=question_text))

        # 4. 流式生成：直接用 OpenAI SDK 的 astream_chat，实时 yield
        async for event, text in astream_chat(messages, thinking=thinking):
            if event == "reasoning":
                reasoning_parts.append(text)
                yield ("reasoning", text)
            elif event == "content":
                answer_parts.append(text)
                yield ("token", text)

        yield ("answer_final", {
            "answer": "".join(answer_parts),
            "reasoning": "".join(reasoning_parts),
        })
        yield ("done", {"status": "ok"})
    except Exception as e:
        yield ("error", {"message": str(e)})
