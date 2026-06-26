"""文本分块策略模块。

支持四种可配置切换的策略（通过 settings.split_strategy 选择）：
  - auto:      按文件类型自动路由（md→markdown，pdf/docx→recursive），默认
  - fixed:     定长字符滑窗（带重叠），最简单快速，不感知语义边界，作为兜底
  - markdown:  先按 Markdown 标题(#/##/###) 切分子节保持结构，再对超长节递归切分
  - recursive: 递归尝试分隔符（段落→换行→句号→空格）切分，兼顾语义边界与长度控制

设计权衡：
  fixed 最快但可能在句子/表格中间硬切；markdown 保结构但依赖文档有标题；
  recursive 通用性最强。auto 按文件类型选最优：原生 md 文件标题结构最完整走 markdown，
  pdf/docx 经 pymupdf4llm/MarkItDown 转换后标题层级可能不规整走 recursive 更稳健。
  fixed 不作为自动选项，保留为手动可配置的兜底。
"""

from langchain_text_splitters import (
    HTMLSemanticPreservingSplitter,
    MarkdownHeaderTextSplitter,
    RecursiveCharacterTextSplitter,
)

# 中文友好的递归分隔符：优先按段落切，其次换行、句号、英文句号、空格
# 含 </table>：MinerU 输出的 HTML 表格在递归切分时优先在表格结束处断开。
# 对不超过 chunk_size 的表格可整体保留；超大表格仍会被切断（递归降级到次级分隔符）。
# 彻底保护整个表格需用 HTMLSemanticPreservingSplitter（已记录为项目演进方向）。
_RECURSIVE_SEPARATORS = ["\n\n", "</table>", "\n", "。", "！", "？", ".", "!", "?", " ", ""]

# Markdown 标题层级 → metadata key 映射
_MD_HEADERS = [("#", "h1"), ("##", "h2"), ("###", "h3")]

# auto 策略的文件类型 → 策略映射
_AUTO_STRATEGY_MAP = {
    "md": "markdown",          # 原生 Markdown，标题结构最完整
    "pdf": "html_preserve",    # MinerU 输出 HTML 表格，用表格保护避免超大表格被切
    "docx": "recursive",       # MarkItDown 转 GFM 表格（非 HTML），走 recursive
}


def _resolve_auto_strategy(ext: str | None) -> str:
    """auto 策略：按文件扩展名路由到具体策略。

    仅支持 md/pdf/docx 三种类型（由 SUPPORTED_EXTS 约束，上传层已拦截其他类型）。
    """
    return _AUTO_STRATEGY_MAP[(ext or "").lower()]


def split_text(
    text: str,
    strategy: str = "recursive",
    chunk_size: int = 500,
    chunk_overlap: int = 100,
    ext: str | None = None,
) -> list[str]:
    """根据策略切分文本，返回非空 chunks 列表。

    Args:
        text: 待分块的原始文本（通常是解析后的 Markdown）
        strategy: auto | fixed | markdown | recursive
        chunk_size: 每块最大字符数
        chunk_overlap: 相邻块重叠字符数
        ext: 文件扩展名（仅 auto 策略用于路由，如 "md"/"pdf"/"docx"）
    """
    if not text or not text.strip():
        return []

    # auto 按文件类型路由
    if strategy == "auto":
        strategy = _resolve_auto_strategy(ext)

    if strategy == "fixed":
        chunks = _fixed_size_split(text, chunk_size, chunk_overlap)
    elif strategy == "markdown":
        chunks = _markdown_split(text, chunk_size, chunk_overlap)
    elif strategy == "html_preserve":
        chunks = _html_preserve_split(text, chunk_size, chunk_overlap)
    else:  # recursive（默认，未知策略也走 recursive）
        chunks = _recursive_split(text, chunk_size, chunk_overlap)

    return [c for c in chunks if c.strip()]


def _fixed_size_split(text: str, chunk_size: int, overlap: int) -> list[str]:
    """定长字符滑窗分块（带重叠）。

    最简单快速：按固定字符数切片，步长 = chunk_size - overlap。
    缺点是不感知任何语义边界，可能在句子/表格中间硬切。
    """
    chunks = []
    start = 0
    step = max(chunk_size - overlap, 1)
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        if end >= len(text):
            break
        start += step
    return chunks


def _recursive_split(text: str, chunk_size: int, overlap: int) -> list[str]:
    """递归字符分块。

    依次尝试 separators 列表中的分隔符，优先在高层级边界（段落）切；
    若某块仍超长，降级用下一级分隔符切，直到每块 ≤ chunk_size。
    兼顾语义边界（尽量不切断句子）与长度控制，通用性强。
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=overlap,
        separators=_RECURSIVE_SEPARATORS,
        keep_separator=True,  # 保留分隔符（如句号），避免块开头/结尾残缺
    )
    return splitter.split_text(text)


def _html_preserve_split(text: str, chunk_size: int, overlap: int) -> list[str]:
    """HTML 语义保留分块（硬保护表格不被切）。

    用 HTMLSemanticPreservingSplitter 的 elements_to_preserve=["table"]，
    硬保证表格整体保留在单一 chunk 内（即使超过 max_chunk_size 也不切）。
    解决 recursive 的 `</table>` 分隔符只能保护中小表格、超大跨页表格仍被切的短板。

    适用场景：PDF 经 MinerU 解析后输出 HTML <table>（保留 rowspan/colspan）。
    DOCX 是 GFM 管道表格（非 HTML），不适用本策略。

    关键 fallback：纯文本无 <table 标签时，HTMLSemanticPreservingSplitter 会返回空列表
    （BeautifulSoup 解析纯文本 body 为 None），导致内容全丢。故检测到无 <table 时
    降级走 recursive，保证无表格文档正常分块。

    注意：保留的是表格的文本内容（get_text 串联），不保留 <tr><td> 标签结构。
    但表格数据完整，LLM 能理解行列关系。如需保留 HTML 结构可用 custom_handlers 迭代。
    """
    # 无 <table → fallback recursive（否则空列表丢全部内容）
    if "<table" not in text:
        return _recursive_split(text, chunk_size, overlap)

    splitter = HTMLSemanticPreservingSplitter(
        headers_to_split_on=_MD_HEADERS,  # 必填，按 #/##/### 标题切节
        max_chunk_size=chunk_size,        # 软上限：为保表格允许超出
        chunk_overlap=overlap,
        elements_to_preserve=["table"],   # 表格整体保留，即使超 max_chunk_size 也不切
    )
    docs = splitter.split_text(text)
    # split_text 返回 list[Document]，适配成 list[str]
    return [d.page_content for d in docs if d.page_content and d.page_content.strip()]


def _markdown_split(text: str, chunk_size: int, overlap: int) -> list[str]:
    """Markdown 标题感知两阶段分块。

    阶段一：MarkdownHeaderTextSplitter 按 #/##/### 标题切成语义完整的"节"，
            每节内容属于同一标题下，保持章节完整性。
    阶段二：对超过 chunk_size 的节，用 RecursiveCharacterTextSplitter 二次切分，
            控制长度同时尽量保句子边界。

    对无标题的纯文本会退化为单节 → 走递归切分（安全降级）。
    """
    try:
        md_splitter = MarkdownHeaderTextSplitter(headers_to_split_on=_MD_HEADERS)
        sections = md_splitter.split_text(text)
    except Exception:
        # 解析失败（非 Markdown 或异常）→ 退化为递归切分
        return _recursive_split(text, chunk_size, overlap)

    if not sections:
        return _recursive_split(text, chunk_size, overlap)

    rc_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=overlap,
        separators=_RECURSIVE_SEPARATORS,
        keep_separator=True,
    )

    chunks = []
    for section in sections:
        # section.page_content 可能含标题文本本身，保留它作为上下文
        content = section.page_content
        if not content or not content.strip():
            continue
        chunks.extend(rc_splitter.split_text(content))
    return chunks
