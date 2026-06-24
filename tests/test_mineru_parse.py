"""MinerU PDF 解析测试。

覆盖：
1. 回退逻辑：无 token → 走 pymupdf4llm；MinerU 失败 → 回退 pymupdf4llm
2. 端到端：有 token 时用 tests/综合篇.pdf 测 MinerU 真实解析（需网络+token，无则 skip）
"""
import os
from pathlib import Path
from unittest.mock import patch

import pytest

from app.services import document_service

TEST_PDF = Path(__file__).parent / "综合篇.pdf"


def test_parse_pdf_no_token_falls_back_to_pymupdf():
    """无 mineru_token 时，直接走 pymupdf4llm 回退。"""
    with patch.object(document_service.settings, "mineru_token", ""):
        # spy: 确认走的是 pymupdf 而非 mineru
        with patch.object(document_service, "_parse_pdf_mineru_sync") as mock_mineru, \
             patch.object(document_service, "_parse_pdf_pymupdf_sync", return_value="pymupdf result") as mock_pymupdf:
            result = document_service._parse_pdf_sync(str(TEST_PDF))
            assert result == "pymupdf result"
            mock_mineru.assert_not_called()
            mock_pymupdf.assert_called_once()


def test_parse_pdf_mineru_failure_falls_back():
    """MinerU 抛异常时，自动回退 pymupdf4llm，不向上传播异常。"""
    with patch.object(document_service.settings, "mineru_token", "fake-token"):
        with patch.object(document_service, "_parse_pdf_mineru_sync", side_effect=RuntimeError("API 挂了")), \
             patch.object(document_service, "_parse_pdf_pymupdf_sync", return_value="fallback ok") as mock_pymupdf:
            result = document_service._parse_pdf_sync(str(TEST_PDF))
            assert result == "fallback ok"
            mock_pymupdf.assert_called_once()


def test_parse_pdf_mineru_timeout_falls_back():
    """MinerU 轮询超时时，回退 pymupdf4llm。"""
    with patch.object(document_service.settings, "mineru_token", "fake-token"):
        with patch.object(document_service, "_parse_pdf_mineru_sync", side_effect=TimeoutError("轮询超时")), \
             patch.object(document_service, "_parse_pdf_pymupdf_sync", return_value="fallback ok") as mock_pymupdf:
            result = document_service._parse_pdf_sync(str(TEST_PDF))
            assert result == "fallback ok"


@pytest.mark.skipif(
    not document_service.settings.mineru_token or not TEST_PDF.exists(),
    reason="需要 mineru_token 配置 + tests/综合篇.pdf 存在",
)
def test_parse_pdf_mineru_e2e():
    """端到端：用真实 MinerU API 解析综合篇.pdf（需 token + 网络）。

    验证：MinerU 输出非空 Markdown，且比 pymupdf4llm 多了图片引用（![](images/)）。
    """
    result = document_service._parse_pdf_mineru_sync(str(TEST_PDF))
    assert len(result) > 100, "MinerU 输出过短，可能解析失败"
    # MinerU 应该能提取图片引用（pymupdf4llm 输出的是 'intentionally omitted'）
    has_img_ref = "![](images/" in result or "![" in result
    assert has_img_ref, "MinerU 输出无图片引用，与预期不符"
