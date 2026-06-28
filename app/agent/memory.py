"""会话记忆工具：token 估算 + 预算截断。

历史消息传给 LLM 前，需要控制总量：
- 纯按轮数截断会让长答案历史迅速撑爆 context（RAG 回答动辄 2-4k token）；
- 这里用「token 预算 + 轮数」双约束，从最新消息向前累加，超任一上限即停，
  保留最新的历史（对多轮指代消解与上下文连贯最重要）。

token 计数用启发式估算（不加 tiktoken 依赖）：
- CJK（中日韩）字符 ≈ 1.5 字/token；
- 其余 ASCII ≈ 4 字符/token（与 OpenAI 英文经验值一致）。
预算控制这种场景下，估算偏差 ±30% 都无所谓——预算 3500 实际 2800-4200 都行。
"""

import unicodedata


def estimate_tokens(text: str) -> int:
    """启发式估算文本的 token 数。

    CJK 字符按 1.5 字/token（中文为主的内容偏保守，避免低估撑爆 context），
    其余字符按 4 字符/token（与 OpenAI 英文 tokenizer 经验值一致）。
    """
    if not text:
        return 0
    cjk = 0
    other = 0
    for ch in text:
        if _is_cjk(ch):
            cjk += 1
        else:
            other += 1
    # CJK: 1.5 字/token → token = cjk/1.5；非 CJK: 4 字符/token → other/4
    return int(cjk / 1.5 + other / 4) + 1  # +1 兜底，避免极短文本算成 0


def _is_cjk(ch: str) -> bool:
    """判断字符是否属于 CJK（中日韩）常用区。

    用 unicodedata.name 探测：CJK 统一表意、平假名/片假名、全角符号等名字含 'CJK'/'HIRAGANA'/'KATAKANA'/'HANGUL'。
    比 hardcode 码段更鲁棒，且对中文为主的文档问答场景覆盖足够。
    """
    try:
        name = unicodedata.name(ch, "")
    except ValueError:
        return False
    return any(tag in name for tag in ("CJK", "HIRAGANA", "KATAKANA", "HANGUL"))


def truncate_by_token_budget(
    history: list[dict], budget: int, max_rounds: int
) -> list[dict]:
    """按 token 预算 + 轮数双约束截断历史，返回正序子集（最旧在前）。

    从最新消息向前累加 token：
    - 累计 token 超过 budget 即停（不在塞入更老的消息）；
    - 已选消息对数超过 max_rounds*2 也停（轮数硬上限，防止预算设得过大）。

    返回的是正序子集（最旧在前），可直接拼进 prompt。
    """
    if not history:
        return []
    selected: list[dict] = []
    used = 0
    # 从最新向前遍历
    for item in reversed(history):
        content = item.get("content", "") or ""
        n = estimate_tokens(content)
        if used + n > budget:
            break
        # 轮数硬上限：已选够 max_rounds 轮就不再往更老塞（selected 是逆序，长度=消息条数）
        if len(selected) >= max_rounds * 2:
            break
        used += n
        selected.append(item)
    # selected 是从新到旧收集的，翻转回正序（最旧在前）
    selected.reverse()
    return selected
