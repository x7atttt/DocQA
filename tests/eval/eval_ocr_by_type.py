"""OCR→RAG 端到端评测：按文档类型分层报告。

复用 run_eval_ocr.py 已建的 chroma_eval_ocr 库和 ocr_md_cache，
按 doc_type 分组统计 Hit Rate，区分"电子版"与"扫描难题"。

电子版（MinerU 直接提取文字，近无损）: font / long / wide / normal*
扫描难题（需 OCR，有识别损耗）: handwriting / high_pixel / watermark / colourful_background / history_book

用法：
    python tests/eval/eval_ocr_by_type.py
"""

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

QA_PATH = os.path.join(os.path.dirname(__file__), "..", "..", "InduOCRBench", "RAG_eval", "QA_pairs.jsonl")
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

# 文档类型分层
ELECTRONIC = {"font", "long", "wide"}  # 电子版 PDF（MinerU 直接提取文字）
SCANNED = {  # 扫描/难题（需 OCR，有损耗）
    "handwriting",
    "high_pixel",
    "watermark",
    "colourful_background",
    "history_book",
    "style",
    "multi_column",
    "cross_page_table",
}


def is_hit(retrieved_chunks, evidence, question):
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
        if ev_compact[:30] in "".join(chunk.split()):
            return True
    return False


async def retrieve_top5(col, question):
    vec = await encode_single(question)
    res = col.query(query_embeddings=[vec], n_results=settings.retrieve_top_k)
    docs = res["documents"][0]
    pairs = await rerank(question, docs, top_k=5)
    return [docs[i] for i, _ in pairs]


async def main():
    print("=" * 60)
    print("OCR→RAG 端到端评测：按文档类型分层")
    print("=" * 60)

    qa_all = [json.loads(l) for l in open(QA_PATH, encoding="utf-8")]
    friendly = [it for it in qa_all if it.get("question_category") in FRIENDLY_CATEGORIES]
    random.seed(42)
    sample = random.sample(friendly, 50)

    # 只保留有缓存解析结果的文档对应的题
    cache_dir = os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..", "..", "data", "ocr_md_cache")
    )
    cached = {f.replace(".md", "") for f in os.listdir(cache_dir) if f.endswith(".md")}
    sample = [q for q in sample if q["filename"].replace(".md", "") in cached]
    print(f"有效题目: {len(sample)} 题（有 MinerU 解析缓存）\n")

    col = chromadb.PersistentClient(path=EVAL_CHROMA_DIR).get_collection(COLLECTION_NAME)

    # 按 doc_type 分桶
    groups = {"electronic": [], "scanned": []}
    for item in sample:
        dt = item.get("doc_type", "")
        if dt in ELECTRONIC:
            groups["electronic"].append(item)
        else:
            groups["scanned"].append(item)

    print(f"电子版(font/long/wide): {len(groups['electronic'])} 题")
    print(f"扫描难题(handwriting/high_pixel等): {len(groups['scanned'])} 题\n")

    results = {}
    for gname, items in groups.items():
        if not items:
            continue
        ks = [1, 3, 5]
        hits = {k: 0 for k in ks}
        mrr = 0.0
        n = len(items)
        print(f"[{gname}] 评测 {n} 题...")
        for idx, item in enumerate(items):
            top5 = await retrieve_top5(col, item["question"])
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
            if (idx + 1) % 5 == 0:
                print(f"    {gname} progress: {idx+1}/{n}")
        results[gname] = {
            "n": n,
            "Hit@1": round(hits[1] / n, 4),
            "Hit@3": round(hits[3] / n, 4),
            "Hit@5": round(hits[5] / n, 4),
            "MRR": round(mrr / n, 4),
        }
        print(f"  结果: {results[gname]}\n")

    # 总体
    total_n = sum(r["n"] for r in results.values())
    print("=" * 60)
    print("分层结果汇总")
    print("=" * 60)
    print(f"{'文档类型':<16} | {'题数':>4} | {'Hit@1':>6} | {'Hit@3':>6} | {'Hit@5':>6} | {'MRR':>6}")
    print("-" * 60)
    for gname in ["electronic", "scanned"]:
        if gname in results:
            r = results[gname]
            label = "电子版(font/long/wide)" if gname == "electronic" else "扫描难题(OCR)"
            print(f"{label:<16} | {r['n']:>4} | {r['Hit@1']:>6} | {r['Hit@3']:>6} | {r['Hit@5']:>6} | {r['MRR']:>6}")
    # 加权平均
    w_hit3 = sum(results[g]["Hit@3"] * results[g]["n"] for g in results) / total_n if total_n else 0
    print(f"{'加权平均':<16} | {total_n:>4} | {'':>6} | {w_hit3:>6.2f} | {'':>6} | {'':>6}")
    print(f"\n（对比：标准标注整体 Hit@3 = 0.94）")

    out = os.path.join(os.path.dirname(__file__), "eval_result_ocr_by_type.json")
    json.dump(results, open(out, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    print(f"\n结果已保存: {out}")


if __name__ == "__main__":
    asyncio.run(main())
