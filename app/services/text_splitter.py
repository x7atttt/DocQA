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
    MarkdownHeaderTextSplitter,
    RecursiveCharacterTextSplitter,
)

# 中文友好的递归分隔符：优先按段落切，其次换行、句号、英文句号、空格
_RECURSIVE_SEPARATORS = ["\n\n", "\n", "。", "！", "？", ".", "!", "?", " ", ""]

# Markdown 标题层级 → metadata key 映射
_MD_HEADERS = [("#", "h1"), ("##", "h2"), ("###", "h3")]

# auto 策略的文件类型 → 策略映射
_AUTO_STRATEGY_MAP = {
    "md": "markdown",     # 原生 Markdown，标题结构最完整
    "pdf": "recursive",   # pymupdf4llm 转换的 md 标题层级可能不规整，recursive 更稳健
    "docx": "recursive",  # MarkItDown 转换的同理
}


def _resolve_auto_strategy(ext: str | None) -> str:
    """auto 策略：按文件扩展名路由到具体策略。"""
    if ext and ext.lower() in _AUTO_STRATEGY_MAP:
        return _AUTO_STRATEGY_MAP[ext.lower()]
    return "recursive"  # 未知类型兜底


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
