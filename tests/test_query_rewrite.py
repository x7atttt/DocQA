"""Query 改写节点单元测试。

验证 rewrite_query 节点的三种行为：
1. 有历史时调 LLM 改写指代词 → rewritten_query 正确赋值
2. 空历史时跳过改写（省调用）→ rewritten_query == question
3. LLM 异常时降级 → rewritten_query == question（不阻断流程）

通过 mock get_llm 控制 LLM 返回，不依赖真实 API。
"""

from unittest.mock import AsyncMock, patch

import pytest

from app.agent.nodes import rewrite_query
from app.agent.state import AgentState


def _make_state(question: str, history: list | None = None) -> AgentState:
    return {
        "user_id": 1,
        "question": question,
        "history": history or [],
        "should_retrieve": True,
    }


@pytest.mark.asyncio
async def test_rewrite_with_history_resolves_coreference():
    """有历史时，LLM 应被调用，指代词被消解成完整问题。"""
    history = [
        {"role": "user", "content": "文档里的规则有哪些？"},
        {"role": "assistant", "content": "文档包含5条规则：1.安全 2.备份 3.加密..."},
    ]
    state = _make_state("第三条是什么？", history)

    # mock LLM：返回改写后的问题
    mock_llm = AsyncMock()
    mock_llm.ainvoke = AsyncMock(return_value=type("R", (), {"content": "文档里第3条规则的具体内容是什么？"})())
    with patch("app.agent.nodes.get_llm", return_value=mock_llm):
        result = await rewrite_query(state)

    assert result["rewritten_query"] == "文档里第3条规则的具体内容是什么？"
    assert result["question"] == "第三条是什么？"  # 原始问题不变
    assert mock_llm.ainvoke.called  # 确实调了 LLM


@pytest.mark.asyncio
async def test_rewrite_empty_history_skips_llm_call():
    """空历史时不调 LLM（省一次调用），rewritten_query == question。"""
    state = _make_state("文档里的规则有哪些？", history=[])

    mock_llm = AsyncMock()
    with patch("app.agent.nodes.get_llm", return_value=mock_llm):
        result = await rewrite_query(state)

    assert result["rewritten_query"] == "文档里的规则有哪些？"
    assert not mock_llm.ainvoke.called  # 没调 LLM


@pytest.mark.asyncio
async def test_rewrite_llm_exception_falls_back_to_question():
    """LLM 异常时降级用原始 question，不阻断流程。"""
    history = [{"role": "user", "content": "上一轮问题"}]
    state = _make_state("它是什么？", history)

    mock_llm = AsyncMock()
    mock_llm.ainvoke = AsyncMock(side_effect=Exception("API 超时"))
    with patch("app.agent.nodes.get_llm", return_value=mock_llm):
        result = await rewrite_query(state)

    # 降级：用原始问题，不抛异常
    assert result["rewritten_query"] == "它是什么？"


@pytest.mark.asyncio
async def test_rewrite_strips_quotes_and_punctuation():
    """改写结果应去除首尾引号/句号（LLM 有时会加引号包裹）。"""
    history = [{"role": "user", "content": "刚才提到的加密"}]
    state = _make_state("具体说说", history)

    # LLM 返回带首尾引号和句号
    fake_resp = type("R", (), {"content": "\u201c加密的具体实现方式\u201d。"})()
    mock_llm = AsyncMock()
    mock_llm.ainvoke = AsyncMock(return_value=fake_resp)
    with patch("app.agent.nodes.get_llm", return_value=mock_llm):
        result = await rewrite_query(state)

    assert result["rewritten_query"] == "加密的具体实现方式"
