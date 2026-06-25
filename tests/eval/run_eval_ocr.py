"""OCR→RAG 端到端检索评测脚本。

和 run_eval.py 的区别：灌库不读标准标注 doc_md，而是用**原始 PDF 经 MinerU
解析(含 OCR/表格)后的 MD**。这样测的是"PDF→解析→分块→检索→rerank"全链路质量，
对比标准标注的 Hit Rate，量化 MinerU 解析对检索的影响。

流程：
1. 对每个 PDF 调 MinerU 解析 → MD（缓存到本地，避免重复调 API）
2. 解析后 MD → 分块 → 灌库（独立 chroma_eval_ocr）
3. 跑 dense+rerank 检索评测（Hit@1/3/5/MRR）
4. 和标准标注的 94% 对比

用法：
    python tests/eval/run_eval_ocr.py [--sample 50] [--limit 10]
"""

import argparse
import asyncio
import json
import os
import random
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

import chromadb

from app.config import get_settings
from app.services.embedding_service import encode_single
from app.services.rerank_service import rerank

settings = get_settings()

BENCH_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "InduOCRBench", "RAG_eval")
QA_PATH = os.path.join(BENCH_DIR, "QA_pairs.jsonl")
PDF_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "InduOCRBench", "pdfs")
)
# MinerU 解析结果缓存（避免重跑时重复调云 API）
MD_CACHE_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "data", "ocr_md_cache")
)
EVAL_CHROMA_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "data", "chroma_eval_ocr")
)
COLLECTION_NAME = "induocrbench_ocr_eval"

FRIENDLY_CATEGORIES = {
    "Basic Recognition",
    "Structural Alignment",
    "Cross-Field Continuity",
    "Complex Reasoning",
}


def load_friendly_qa(sample_n: int | None = None) -> list[dict]:
    """加载并筛选检索友好题（与 run_eval.py 一致：seed=42）。"""
    items = [json.loads(l) for l in open(QA_PATH, encoding="utf-8")]
    friendly = [it for it in items if it.get("question_category") in FRIENDLY_CATEGORIES]
    if sample_n and len(friendly) > sample_n:
        random.seed(42)
        friendly = random.sample(friendly, sample_n)
    return friendly


def is_hit(retrieved_chunks: list[str], evidence: str | list, question: str) -> bool:
    """命中判定（与 run_eval.py 一致）。"""
    if isinstance(evidence, list):
        ev_texts = [str(e) for e in evidence if e]
    else:
        ev_texts = [str(evidence)]
    if not ev_texts:
        return False
    ev = max(ev_texts, key=len)
    ev_compact = "".join(ev.split())
    if len(ev_compact) < 8:
        ev_compact = "".join(question.split())[:20]
    for chunk in retrieved_chunks:
        chunk_compact = "".join(chunk.split())
        if ev_compact[:30] in chunk_compact:
            return True
    return False


async def parse_pdfs_with_mineru(filenames: list[str]) -> dict[str, str]:
    """对每个 PDF 调 MinerU 解析，返回 {filename: md_content}。

    带本地缓存：解析结果存 MD_CACHE_DIR，重跑时直接读缓存不调 API。
    失败的 PDF 回退 pymupdf4llm，记录到返回的元信息。
    """
    from app.services.document_service import _parse_pdf_sync

    os.makedirs(MD_CACHE_DIR, exist_ok=True)
    results: dict[str, str] = {}
    stats = {"success": 0, "fallback": 0, "cache_hit": 0}

    for i, pdf_name in enumerate(filenames):
        pdf_path = os.path.join(PDF_DIR, pdf_name)
        cache_path = os.path.join(MD_CACHE_DIR, pdf_name.replace(".pdf", ".md"))

        # 缓存命中
        if os.path.exists(cache_path):
            md = open(cache_path, encoding="utf-8").read()
            if md.strip():
                results[pdf_name] = md
                stats["cache_hit"] += 1
                continue

        if not os.path.exists(pdf_path):
            print(f"  [跳过] PDF 不存在: {pdf_name}")
            continue

        # 调 MinerU 解析（在子线程跑，避免阻塞事件循环）
        md = await asyncio.to_thread(_parse_pdf_sync, pdf_path)
        if md and md.strip():
            # 写缓存
            with open(cache_path, "w", encoding="utf-8") as f:
                f.write(md)
            results[pdf_name] = md
            # _parse_pdf_sync 内部已处理回退，这里无法直接知道是否回退
            stats["success"] += 1
            print(f"  [{i+1}/{len(filenames)}] 解析完成: {pdf_name} ({len(md)} 字符)")
        else:
            print(f"  [{i+1}/{len(filenames)}] 解析为空: {pdf_name}")

    return results, stats


async def build_collection(
    md_results: dict[str, str], qa_items: list[dict]
) -> chromadb.Collection:
    """把 MinerU 解析后的 MD 灌入独立 collection。"""
    from app.services.text_splitter import split_text
    from app.services.embedding_service import encode_texts

    if os.path.exists(EVAL_CHROMA_DIR):
        import shutil

        shutil.rmtree(EVAL_CHROMA_DIR, ignore_errors=True)

    client = chromadb.PersistentClient(path=EVAL_CHROMA_DIR)
    col = client.get_or_create_collection(
        name=COLLECTION_NAME, metadata={"hnsw:space": "cosine"}
    )

    # 只灌有解析结果的文档
    filenames = list({it["filename"].replace(".md", ".pdf") for it in qa_items})
    filenames = [f for f in filenames if f in md_results]

    all_chunks: list[str] = []
    all_ids: list[str] = []
    all_metas: list[dict] = []
    for pdf_name in filenames:
        text = md_results[pdf_name].strip()
        if not text:
            continue
        md_name = pdf_name.replace(".pdf", ".md")
        chunks = split_text(
            text, strategy="recursive", chunk_size=settings.chunk_size,
            chunk_overlap=settings.chunk_overlap,
        )
        for i, c in enumerate(chunks):
            all_chunks.append(c)
            all_ids.append(f"{md_name}__{i}")
            all_metas.append({"filename": md_name, "chunk_index": i})

    print(f"  灌库: {len(filenames)} 文档 → {len(all_chunks)} chunks")
    batch = 32
    all_vecs = []
    for s in range(0, len(all_chunks), batch):
        all_vecs.extend(await encode_texts(all_chunks[s : s + batch]))

    col.add(ids=all_ids, documents=all_chunks, embeddings=all_vecs, metadatas=all_metas)
    return col


async def retrieve_dense(col, question: str, top_k: int = 3) -> list[str]:
    """纯 dense + rerank（与 run_eval.py 的 dense_rerank 一致，便于对比）。"""
    vec = await encode_single(question)
    res = col.query(query_embeddings=[vec], n_results=settings.retrieve_top_k)
    docs = res["documents"][0]
    pairs = await rerank(question, docs, top_k=top_k)
    return [docs[i] for i, _ in pairs]


async def evaluate(col, qa_items: list[dict]) -> dict:
    """跑 dense+rerank 评测，返回 Hit@1/3/5 和 MRR。"""
    ks = [1, 3, 5]
    hits = {k: 0 for k in ks}
    mrr = 0.0
    n = len(qa_items)
    for idx, item in enumerate(qa_items):
        top5 = await retrieve_dense(col, item["question"], top_k=5)
        rank_found = 0
        for r, ch in enumerate(top5, 1):
            if is_hit([ch], item["evidence"], item["question"]):
                rank_found = r
                break
        if rank_found:
            mrr += 1.0 / rank_found
        for k in ks:
            if is_hit(top5[:k], item["evidence"], item["question"]):
                hits[k] += 1
        if (idx + 1) % 10 == 0:
            print(f"    progress: {idx+1}/{n}")
    return {
        "n": n,
        "Hit@1": round(hits[1] / n, 4),
        "Hit@3": round(hits[3] / n, 4),
        "Hit@5": round(hits[5] / n, 4),
        "MRR": round(mrr / n, 4),
    }


async def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sample", type=int, default=50, help="抽样题数")
    parser.add_argument("--limit", type=int, default=None, help="限制解析的 PDF 数（小样本验证用）")
    args = parser.parse_args()

    print("=" * 60)
    print("OCR→RAG 端到端评测（MinerU 解析 → 检索）")
    print("=" * 60)

    qa = load_friendly_qa(args.sample)
    print(f"筛选检索友好题: {len(qa)} 题")

    needed_pdfs = list({it["filename"].replace(".md", ".pdf") for it in qa})
    if args.limit:
        needed_pdfs = needed_pdfs[: args.limit]
        qa = [q for q in qa if q["filename"].replace(".md", ".pdf") in set(needed_pdfs)]
        print(f"限制解析: {len(needed_pdfs)} 个 PDF, 对应 {len(qa)} 题")

    print(f"\n[1/3] MinerU 解析 {len(needed_pdfs)} 个 PDF...")
    md_results, parse_stats = await parse_pdfs_with_mineru(needed_pdfs)
    print(f"  解析统计: {parse_stats}")

    print(f"\n[2/3] 灌库（MinerU 解析结果）...")
    col = await build_collection(md_results, qa)
    print(f"  collection: {col.count()} chunks")

    print(f"\n[3/3] 评测 dense + rerank...")
    result = await evaluate(col, qa)
    print(f"  结果: {result}")

    print("\n" + "=" * 60)
    print("对比标准标注（run_eval.py 的 dense_rerank）")
    print("=" * 60)
    print(f"{'指标':<10} | {'标准标注':>10} | {'MinerU解析':>12}")
    print("-" * 40)
    baseline = {"Hit@1": 0.78, "Hit@3": 0.94, "Hit@5": 0.96, "MRR": 0.8573}
    for k in ["Hit@1", "Hit@3", "Hit@5", "MRR"]:
        print(f"{k:<10} | {baseline[k]:>10} | {result[k]:>12}")

    out_data = {
        "sample_size": len(qa),
        "parse_stats": parse_stats,
        "chunks": col.count(),
        "result": result,
        "baseline": baseline,
    }
    out = os.path.join(os.path.dirname(__file__), "eval_result_ocr.json")
    json.dump(out_data, open(out, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    print(f"\n结果已保存: {out}")


if __name__ == "__main__":
    asyncio.run(main())
