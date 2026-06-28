"""会话记忆工具单元测试。

验证 estimate_tokens 与 truncate_by_token_budget：
1. token 估算对中文/英文分别合理（CJK 按 1.5 字/token，其余 4 字符/token）
2. token 预算截断：超预算时从最老开始丢弃，保留最新
3. 轮数上限：未超预算但超轮数时按轮数截
4. 截断后保留的总是最新几条（顺序正确）

纯函数测试，不依赖 LLM / DB / Redis。
"""

from app.agent.memory import estimate_tokens, truncate_by_token_budget


def test_estimate_tokens_empty():
    """空字符串 token 数应为 0。"""
    assert estimate_tokens("") == 0


def test_estimate_tokens_cjk():
    """中文字符按 1.5 字/token 估算：3 字 ≈ 2 token（+1 兜底 → 实际 3）。"""
    # 3 个中文：3/1.5 = 2，+1 兜底 = 3
    assert estimate_tokens("你好世") == 3
    # 6 个中文：6/1.5 = 4，+1 = 5
    assert estimate_tokens("你好世界你好") == 5


def test_estimate_tokens_ascii():
    """英文字符按 4 字符/token 估算：8 字符 ≈ 2 token（+1 兜底 → 实际 3）。"""
    # 8 个 ascii：8/4 = 2，+1 兜底 = 3
    assert estimate_tokens("abcdefgh") == 3


def test_estimate_tokens_mixed():
    """中英混合：中文按 1.5 字/token、英文按 4 字符/token 分别累加。"""
    # "你好"(2 cjk) + "ab"(2 ascii)：2/1.5 + 2/4 = 1.33 + 0.5 = 1.83 → int=1 +1 = 2
    assert estimate_tokens("你好ab") == 2


def test_truncate_empty_history():
    """空历史截断后仍为空。"""
    assert truncate_by_token_budget([], budget=1000, max_rounds=5) == []


def test_truncate_within_limits_returns_all():
    """未超预算也未超轮数时返回全部历史。"""
    history = [
        {"role": "user", "content": "问题1"},
        {"role": "assistant", "content": "回答1"},
        {"role": "user", "content": "问题2"},
        {"role": "assistant", "content": "回答2"},
    ]
    result = truncate_by_token_budget(history, budget=10000, max_rounds=5)
    assert result == history  # 原样返回，顺序不变（最旧在前）


def test_truncate_respects_token_budget():
    """超 token 预算时从最老开始丢弃，保留最新。"""
    # 每条 200 个中文字 ≈ 134 token；预算设小，强制只能塞 1-2 条
    long_content = "字" * 200
    history = [
        {"role": "user", "content": long_content},      # 最老
        {"role": "assistant", "content": long_content},
        {"role": "user", "content": long_content},
        {"role": "assistant", "content": long_content},  # 最新
    ]
    # 预算 150 token：最新一条 134 token < 150 能进，第二条累计 268 > 150 停
    result = truncate_by_token_budget(history, budget=150, max_rounds=5)
    # 只保留最新 1 条
    assert len(result) == 1
    assert result[0]["content"] == long_content
    assert result[0]["role"] == "assistant"  # 是最新那条


def test_truncate_respects_max_rounds():
    """未超 token 预算但超轮数上限时，按轮数截断。"""
    history = [
        {"role": "user", "content": "短问题1"},
        {"role": "assistant", "content": "短回答1"},
        {"role": "user", "content": "短问题2"},
        {"role": "assistant", "content": "短回答2"},
        {"role": "user", "content": "短问题3"},       # 最新轮
        {"role": "assistant", "content": "短回答3"},
    ]
    # 预算很大不触发，max_rounds=2 限制只保留最近 2 轮（4 条消息）
    result = truncate_by_token_budget(history, budget=100000, max_rounds=2)
    assert len(result) == 4  # 2 轮 = 4 条
    # 保留的是最新的 2 轮（问题2-回答2、问题3-回答3）
    assert result[0]["content"] == "短问题2"
    assert result[-1]["content"] == "短回答3"


def test_truncate_keeps_latest_and_order():
    """截断后保留的总是最新几条，且正序（最旧在前）。"""
    history = []
    for i in range(10):
        history.append({"role": "user", "content": f"问题{i}"})
        history.append({"role": "assistant", "content": f"回答{i}"})

    # max_rounds=3 → 保留最近 3 轮（6 条）
    result = truncate_by_token_budget(history, budget=100000, max_rounds=3)
    assert len(result) == 6
    # 正序：最旧在前。保留的是第 7/8/9 轮
    assert result[0]["content"] == "问题7"
    assert result[-1]["content"] == "回答9"
    # 顺序正确：user/assistant 交替
    roles = [item["role"] for item in result]
    assert roles == ["user", "assistant", "user", "assistant", "user", "assistant"]
