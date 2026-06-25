import json
from collections.abc import AsyncIterator
from typing import Any

from langgraph.graph.state import CompiledStateGraph

from app.agent.state import AgentState


def sse(event: str, data: Any) -> str:
    payload = json.dumps(data, ensure_ascii=False) if not isinstance(data, str) else data
    return f"event: {event}\ndata: {payload}\n\n"


def _extract_reasoning(chunk: Any) -> str:
    """从流式 chunk 中提取推理增量。

    langchain-deepseek 把 reasoning_content 解析到
    chunk.additional_kwargs['reasoning_content']（见 ChatDeepSeek 源码）。
    这里做兼容性兜底。
    """
    aw = getattr(chunk, "additional_kwargs", None) or {}
    if isinstance(aw, dict):
        rc = aw.get("reasoning_content")
        if isinstance(rc, str) and rc:
            return rc
    rc = getattr(chunk, "reasoning_content", None)
    if isinstance(rc, str) and rc:
        return rc
    rm = getattr(chunk, "response_metadata", None) or {}
    if isinstance(rm, dict):
        rc = rm.get("reasoning_content")
        if isinstance(rc, str) and rc:
            return rc
    return ""


# 哪些节点的 LLM 流才算"正式输出"（intent_router 的 yes/no 不算）
_ANSWER_NODES = {"generate_answer", "general_answer"}


async def stream_graph(
    graph: CompiledStateGraph,
    user_id: int,
    question: str,
    history: list[dict] | None = None,
    thinking: bool = False,
) -> AsyncIterator[tuple[str, Any]]:
    """yield (event_name, payload). event ∈ reasoning/token/sources/done/error.

    统一走 graph.astream_events：
    - 通过 on_chain_start/end 追踪当前节点，只放行 generate_answer/general_answer
      的 LLM 流式 token，过滤掉 intent_router 的 yes/no 判断
    - thinking 开启时，ChatDeepSeek 的推理内容出现在 chunk.additional_kwargs，
      通过 reasoning 事件单独推送（在 token 之前）
    """
    initial: AgentState = {
        "user_id": user_id,
        "question": question,
        "rewritten_query": "",
        "history": history or [],
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
    sources_emitted = False
    # 追踪当前所在节点栈（langgraph 节点名，非 LLM 类名）
    current_chain: list[str] = []

    try:
        async for evt in graph.astream_events(initial, version="v2"):
            kind = evt.get("event")
            name = evt.get("name", "")

            # 维护当前节点栈
            if kind == "on_chain_start" and name:
                current_chain.append(name)
            elif kind == "on_chain_end" and name and current_chain:
                if name in current_chain:
                    idx = len(current_chain) - 1 - current_chain[::-1].index(name)
                    current_chain = current_chain[:idx]

            # 检索完成 → 推送来源
            if kind == "on_chain_end" and name == "retrieve_documents":
                output = evt.get("data", {}).get("output")
                if isinstance(output, dict) and output.get("sources") and not sources_emitted:
                    sources_emitted = True
                    yield ("sources", output["sources"])

            # LLM 流式 token —— 只接受答案节点
            elif kind == "on_chat_model_stream":
                if not any(n in _ANSWER_NODES for n in current_chain):
                    continue
                chunk = evt.get("data", {}).get("chunk")
                if chunk is None:
                    continue

                # 推理内容（reasoning_content）—— 单独事件，在 token 之前
                rc = _extract_reasoning(chunk)
                if rc:
                    reasoning_parts.append(rc)
                    yield ("reasoning", rc)

                # 正式答案
                content = getattr(chunk, "content", None)
                if isinstance(content, str) and content:
                    answer_parts.append(content)
                    yield ("token", content)

        answer = "".join(answer_parts)
        yield ("answer_final", {
            "answer": answer,
            "reasoning": "".join(reasoning_parts),
        })
        yield ("done", {"status": "ok"})
    except Exception as e:
        yield ("error", {"message": str(e)})
