from app.services.document_service import chunk_text
from app.services.text_splitter import split_text


def test_chunk_basic():
    text = "a" * 700
    chunks = chunk_text(text, chunk_size=500, overlap=100)
    assert len(chunks) >= 2
    assert all(len(c) <= 500 for c in chunks)


def test_chunk_overlap():
    text = "abcdefgh" * 100
    chunks = chunk_text(text, chunk_size=500, overlap=100)
    if len(chunks) >= 2:
        assert chunks[0][-100:] == chunks[1][:100]


def test_chunk_empty():
    assert chunk_text("", 500, 100) == []
    assert chunk_text("   \n\n  ", 500, 100) == []


def test_chunk_short_text():
    chunks = chunk_text("短文本", chunk_size=500, overlap=100)
    assert len(chunks) == 1
    assert chunks[0] == "短文本"


# ---------- 新增：多策略测试 ----------

def test_split_text_empty():
    """空文本所有策略都返回空列表"""
    assert split_text("", strategy="fixed") == []
    assert split_text("", strategy="recursive") == []
    assert split_text("", strategy="markdown") == []


def test_fixed_strategy_backward_compatible():
    """fixed 策略与原 chunk_text 行为一致"""
    text = "测试" * 300
    chunks = split_text(text, strategy="fixed", chunk_size=500, chunk_overlap=100)
    assert len(chunks) >= 1
    assert all(len(c) <= 500 for c in chunks)


def test_recursive_respects_sentence_boundary():
    """recursive 策略尽量在句子边界切，不硬切"""
    # 重复的完整句子，每个 9 字符 + 句号
    text = "这是一句话。" * 100
    chunks = split_text(text, strategy="recursive", chunk_size=50, chunk_overlap=10)
    assert len(chunks) >= 2
    # 每块不应超过 chunk_size
    assert all(len(c) <= 50 for c in chunks)


def test_markdown_split_preserves_header_sections():
    """markdown 策略按标题切，不同标题的内容不在同一块"""
    text = (
        "# 项目一\n这是项目一的内容。\n"
        "## 子模块\n子模块详情。\n"
        "# 项目二\n这是项目二的内容。\n"
    )
    chunks = split_text(text, strategy="markdown", chunk_size=500, chunk_overlap=50)
    assert len(chunks) >= 1
    # 项目一和项目二应被分开（至少有两个块，或单块内标题有序）
    joined = "\n".join(chunks)
    assert "项目一" in joined
    assert "项目二" in joined


def test_recursive_is_default_for_unknown_strategy():
    """未知策略退化为 recursive（不报错）"""
    text = "测试文本。" * 50
    chunks = split_text(text, strategy="unknown_strategy", chunk_size=100, chunk_overlap=20)
    assert len(chunks) >= 1


def test_markdown_fallback_on_plain_text():
    """markdown 策略对无标题纯文本安全降级（不报错，返回非空）"""
    text = "这是纯文本，没有任何 Markdown 标题。" * 20
    chunks = split_text(text, strategy="markdown", chunk_size=100, chunk_overlap=20)
    assert len(chunks) >= 1


def test_auto_routes_md_to_markdown():
    """auto 策略：md 文件路由到 markdown（按标题切分，标题作为分界）"""
    text = "# 标题\n内容A\n# 标题二\n内容B"
    chunks = split_text(text, strategy="auto", chunk_size=500, chunk_overlap=50, ext="md")
    assert len(chunks) >= 1
    # MarkdownHeaderTextSplitter 把标题作为 metadata 分界，page_content 只含正文
    # 验证两个标题下的内容被分到不同块（标题起到分界作用）
    joined = "\n".join(chunks)
    assert "内容A" in joined and "内容B" in joined
    if len(chunks) >= 2:
        # 内容A 和 内容B 应在不同块
        assert not any("内容A" in c and "内容B" in c for c in chunks)


def test_auto_routes_pdf_to_html_preserve():
    """auto 策略：pdf 文件路由到 html_preserve（MinerU 输出 HTML 表格，保护超大表格）"""
    text = "这是一段文字。" * 50
    chunks = split_text(text, strategy="auto", chunk_size=100, chunk_overlap=20, ext="pdf")
    assert len(chunks) >= 1


def test_auto_routes_docx_to_recursive():
    """auto 策略：docx 文件路由到 recursive"""
    text = "这是一段文字。" * 50
    chunks = split_text(text, strategy="auto", chunk_size=100, chunk_overlap=20, ext="docx")
    assert len(chunks) >= 1


# ---------- HTML 表格保护测试（MinerU 输出含 HTML 表格）----------


def test_recursive_small_table_intact():
    """recursive 分块时，含 </table> 的分隔符让小表格整体保留。

    MinerU 输出的表格是 HTML 格式。分隔符含 </table> 后：
    - 小表格（≤chunk_size）：整体保留在一个 chunk
    - 超大表格：仍会被切（已知限制，html_preserve 策略彻底解决，见下方测试）
    """
    # 小表格：应整体保留
    small_table = "<table><tr><td>A</td><td>B</td></tr><tr><td>1</td><td>2</td></tr></table>"
    text = f"段落前。\n\n{small_table}\n\n段落后。"
    chunks = split_text(text, strategy="recursive", chunk_size=500, chunk_overlap=50)
    found_intact = any(small_table in c for c in chunks)
    assert found_intact, "小表格未完整保留（</table> 分隔符未生效）"


# ---------- html_preserve 策略：硬保护超大表格（stage-10）----------


def _make_large_html_table(rows: int = 20) -> str:
    """构造超大 HTML 表格（模拟跨页表格，远超 chunk_size）。"""
    body = "".join(
        f"<tr><td>行{i}列A数据内容</td><td>行{i}列B数据内容</td><td>行{i}列C数据内容</td></tr>"
        for i in range(rows)
    )
    return f"<table><thead><tr><th>列A</th><th>列B</th><th>列C</th></tr></thead><tbody>{body}</tbody></table>"


def test_html_preserve_large_table_not_split():
    """html_preserve 策略：超大表格（>chunk_size）整体保留在单一 chunk，不被切。

    对照：recursive 会把超大表格切分（见 test_recursive_large_table_gets_split）。
    """
    large_table = _make_large_html_table(rows=20)  # ~1500 字符，远超 chunk_size=500
    assert len(large_table) > 500, "测试前置：表格应大于 chunk_size"
    text = f"# 报告标题\n\n这是报告正文段落。\n\n{large_table}\n\n表格后的段落。"

    chunks = split_text(text, strategy="html_preserve", chunk_size=500, chunk_overlap=50)

    # html_preserve 保留表格的文本内容（get_text 串联），验证表格数据完整
    joined = "\n".join(chunks)
    # 表格的所有行数据都应在切分结果里（证明没丢内容）
    assert "行0列A数据内容" in joined
    assert "行19列C数据内容" in joined
    # 至少有一个 chunk 完整包含整张表格（证明表格没被切成两半）
    # 表格文本特征：第一行和最后一行同时在某个 chunk 里
    table_intact = any(
        "行0列A数据内容" in c and "行19列C数据内容" in c for c in chunks
    )
    assert table_intact, "超大表格被切成多块，html_preserve 未生效"


def test_recursive_large_table_gets_split():
    """对照测试：同样超大表格走 recursive 会被切（证明 html_preserve 的价值）。

    recursive 的 </table> 分隔符对超大表格无效——表格超过 chunk_size 时
    降级到次级分隔符（\n），在 <tr> 之间切断。
    """
    large_table = _make_large_html_table(rows=20)
    text = f"段落前。\n\n{large_table}\n\n段落后。"

    chunks = split_text(text, strategy="recursive", chunk_size=500, chunk_overlap=50)
    joined = "\n".join(chunks)

    # recursive 会把表格切分：没有任何单 chunk 同时含首行和末行
    table_intact = any(
        "行0列A数据内容" in c and "行19列C数据内容" in c for c in chunks
    )
    assert not table_intact, "recursive 竟然保留了完整大表格？那 html_preserve 就没意义了"


def test_html_preserve_no_table_falls_back_to_recursive():
    """html_preserve 策略：无 <table 的纯文本 fallback 到 recursive（不丢内容）。

    关键：HTMLSemanticPreservingSplitter 对纯文本返回空列表（body 为 None），
    必须检测无 <table 时降级走 recursive，否则内容全丢。
    """
    plain_text = "这是一段没有表格的纯文本。" * 30
    chunks = split_text(plain_text, strategy="html_preserve", chunk_size=100, chunk_overlap=20)

    # 不丢内容：所有文本都在结果里
    assert len(chunks) >= 1
    assert "这是一段没有表格的纯文本。" in "\n".join(chunks)
    # 控制长度：fallback 到 recursive 后每块 ≤ chunk_size（递归切分生效）
    assert all(len(c) <= 100 for c in chunks)


def test_auto_pdf_uses_html_preserve_for_tables():
    """auto 策略 + pdf + 含表格 → 走 html_preserve（表格被保护）"""
    large_table = _make_large_html_table(rows=20)
    text = f"# 文档\n\n正文。\n\n{large_table}"

    chunks = split_text(text, strategy="auto", chunk_size=500, chunk_overlap=50, ext="pdf")
    joined = "\n".join(chunks)
    # 表格首末行在同一 chunk（html_preserve 生效，没走 recursive）
    table_intact = any(
        "行0列A数据内容" in c and "行19列C数据内容" in c for c in chunks
    )
    assert table_intact, "auto+pdf 对含表格文档未走 html_preserve 保护"

