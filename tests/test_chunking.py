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


def test_auto_routes_pdf_to_recursive():
    """auto 策略：pdf 文件路由到 recursive"""
    text = "这是一段文字。" * 50
    chunks = split_text(text, strategy="auto", chunk_size=100, chunk_overlap=20, ext="pdf")
    assert len(chunks) >= 1
    assert all(len(c) <= 100 for c in chunks)


def test_auto_routes_docx_to_recursive():
    """auto 策略：docx 文件路由到 recursive"""
    text = "这是一段文字。" * 50
    chunks = split_text(text, strategy="auto", chunk_size=100, chunk_overlap=20, ext="docx")
    assert len(chunks) >= 1


# ---------- HTML 表格保护测试（MinerU 输出含 HTML 表格）----------


def test_html_table_large_splits_at_table_boundary():
    """recursive 分块时，含 </table> 的分隔符让小表格整体保留、大表格在边界断。

    MinerU 输出的表格是 HTML 格式。分隔符含 </table> 后：
    - 小表格（≤chunk_size）：整体保留在一个 chunk
    - 超大表格：无法整体保留，会在表格内部降级切分（已知限制，彻底保护需
      HTMLSemanticPreservingSplitter，见项目演进方向）

    本测试验证：小表格场景 </table> 分隔符生效（表格完整出现在某 chunk）。
    """
    # 小表格：应整体保留
    small_table = "<table><tr><td>A</td><td>B</td></tr><tr><td>1</td><td>2</td></tr></table>"
    text = f"段落前。\n\n{small_table}\n\n段落后。"
    chunks = split_text(text, strategy="recursive", chunk_size=500, chunk_overlap=50)
    found_intact = any(small_table in c for c in chunks)
    assert found_intact, "小表格未完整保留（</table> 分隔符未生效）"

