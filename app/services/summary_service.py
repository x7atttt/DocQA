"""会话摘要服务：长对话老上下文的滑动窗口压缩。

场景：用户深挖一篇文档时，早期问答（如"论文的核心方法是什么"）在超过
max_history_rounds 后会掉出生成窗口，导致后续"那个方法的对比例…"的指代无法消解。
本服务在累计轮数达 summarize_round_threshold 后，把【窗口外的老对话】压缩成摘要，
注入 system prompt，与近期原文构成"摘要 + 近期原文"的混合长期记忆（Dify/FastGPT 同款方案）。

触发：异步 BackgroundTasks（用户请求不阻塞），用 SETNX 锁防止同一会话重复生成。
降级：LLM 异常 / Redis 不可用都不阻断主流程，只记日志。
"""

import logging

from langchain_core.messages import HumanMessage, SystemMessage
from sqlalchemy import func, select

from app.agent.memory import estimate_tokens
from app.config import get_settings
from app.core.cache import acquire_summary_lock, release_summary_lock
from app.core.database import async_session_factory
from app.models import Conversation, Message
from app.services.llm_provider import chat

settings = get_settings()
logger = logging.getLogger("docqa.summary")


async def maybe_generate_summary(conv_id: int) -> None:
    """达阈值则异步生成会话摘要，写回 Conversation.summary。

    流程：
      1. 统计该会话 message 条数 → 轮数；不足阈值直接返回（省 LLM 调用）
      2. 加 SETNX 锁（按 conv_id 防重复生成）；拿不到锁说明已有 worker 在生成，直接返回
      3. 取【早于最近 max_history_rounds 轮】的老消息（窗口外、需被压缩的部分）
      4. 拼 dialogue 调 chat() 生成摘要
      5. 写回 conv.summary
    全程容错：任何异常只记日志，不影响用户已收到的回答。
    """
    threshold = settings.summarize_round_threshold
    try:
        async with async_session_factory() as db:
            msg_count = await db.scalar(
                select(func.count(Message.id)).where(Message.conversation_id == conv_id)
            )
            # 每轮 = user + assistant 两条消息
            rounds = (msg_count or 0) // 2
            if rounds < threshold:
                return  # 不足阈值，不生成

            # 加锁：拿不到说明已有 worker 在生成，跳过
            lock_token = await acquire_summary_lock(conv_id)
            if lock_token is None:
                logger.info(f"conv_{conv_id} 摘要已在生成，跳过")
                return
            try:
                await _generate_and_persist(db, conv_id)
            finally:
                await release_summary_lock(conv_id, lock_token)
    except Exception as e:
        # 降级：摘要失败绝不影响主流程（用户已拿到答案）
        logger.warning(f"conv_{conv_id} 摘要生成失败（已降级忽略）：{e}")


async def _generate_and_persist(db, conv_id: int) -> None:
    """取窗口外老对话 → 调 LLM 摘要 → 写回 conv.summary。

    单独拆出便于在持锁区间内执行，保证锁释放不遗漏。
    """
    # 取该会话全部消息（正序：最旧在前）
    result = await db.execute(
        select(Message)
        .where(Message.conversation_id == conv_id)
        .order_by(Message.id.asc())
    )
    all_msgs = result.scalars().all()
    if not all_msgs:
        return

    # 窗口外 = 全部去掉最近 max_history_rounds 轮（保留近期原文不压缩）
    window_size = settings.max_history_rounds * 2
    old_msgs = all_msgs[:-window_size] if len(all_msgs) > window_size else []
    if not old_msgs:
        return  # 老对话为空（刚到阈值，全在窗口内），无需压缩

    # 拼 dialogue，并对每条内容做长度截断（防止单条过长撑爆摘要 prompt）
    lines = []
    for m in old_msgs:
        if m.role not in ("user", "assistant") or not m.content:
            continue
        role = "用户" if m.role == "user" else "助手"
        # 单条上限 ~600 token（约 400 中文字 / 2400 英文字符），超出截断
        content = _truncate(m.content, max_tokens=600)
        lines.append(f"{role}: {content}")
    dialogue = "\n".join(lines)
    if not dialogue.strip():
        return

    messages = [
        SystemMessage(
            content=(
                "你是会话摘要助手。把以下多轮对话浓缩成关键事实、实体与上下文，"
                "供后续问答参考（用于消解指代词、保持长对话连贯）。\n"
                "要求：\n"
                "1) 只保留对话中明确出现的事实（用户问了什么、文档涉及哪些主题/实体/结论）；\n"
                "2) 不要编造未出现的内容；\n"
                "3) 用简洁的要点形式，不超过 200 字；\n"
                "4) 如果对话中有明确的文档主题，点出主题关键词。"
            )
        ),
        HumanMessage(content=f"对话内容：\n{dialogue}\n\n摘要："),
    ]
    summary = await chat(messages, max_tokens=settings.summary_max_tokens)
    summary = summary.strip()
    if not summary:
        return

    # 写回 Conversation.summary
    conv = await db.get(Conversation, conv_id)
    if conv is not None:
        conv.summary = summary
        await db.commit()
        logger.info(f"conv_{conv_id} 摘要已生成（{estimate_tokens(summary)} tokens）")


def _truncate(text: str, max_tokens: int) -> str:
    """按 token 预算截断单条内容（粗略：超长则按字符裁剪）。"""
    if estimate_tokens(text) <= max_tokens:
        return text
    # 启发式倒推字符上限：CJK 1.5字/token → max_tokens*1.5 字符兜底
    return text[: int(max_tokens * 1.5)] + "…"
