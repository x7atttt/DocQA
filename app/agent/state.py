from typing import Any, TypedDict


class SourceItem(TypedDict):
    document_id: int
    filename: str
    chunk_index: int
    content: str
    score: float


class HistoryItem(TypedDict):
    """单条历史对话，role ∈ {'user', 'assistant'}。"""
    role: str
    content: str


class AgentState(TypedDict, total=False):
    user_id: int
    question: str
    # query 改写：多轮指代消解后的检索 query（仅检索用，生成仍用原始 question）
    rewritten_query: str
    # 意图路由
    should_retrieve: bool
    # 检索
    retrieved_docs: list[str]
    sources: list[SourceItem]
    # 多轮上下文：最近 N 轮历史（正序，最旧在前）
    history: list[HistoryItem]
    # 会话摘要：长对话老上下文的压缩（达阈值异步生成），注入 system prompt 做长期记忆
    summary: str
    # 是否开启 DeepSeek thinking 模式（用户自定义）
    thinking: bool
    # 答案
    answer_tokens: list[str]
    answer: str
    # 推理内容（DeepSeek reasoner / thinking 模式）
    reasoning_tokens: list[str]
    reasoning: str
    error: str | None
    meta: dict[str, Any]
