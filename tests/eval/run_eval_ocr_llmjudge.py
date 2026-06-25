"""OCR→RAG 端到端评测（LLM-as-judge 判定版）。

和 run_eval_ocr.py 的区别：命中判定从"evidence 精确子串匹配"改为
**LLM-as-judge**（业界主流，RAGAS Context Recall 思路）。

判定方式：对每题，把 question + answer + 检索 Top5 的 chunk 一次性喂给 LLM，
让它逐个判定 chunk 是否"包含回答该问题所需的信息"，返回每个 rank 的 yes/no。
- 消除 OCR 输出与标准标注的格式差异误判（表格 HTML 属性、标点、别字）
- 数字才与标准标注的 94% 有可比性

复用：
- data/chroma_eval_ocr（已建库，626 chunks）
- data/ocr_md_cache（MinerU 解析缓存）

用法：
    python tests/eval/run_eval_ocr_llmjudge.py [--sample 50]
"""

import argparse
import asyncio
import json
import os
import random
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

import chromadb
from langchain_core.messages import HumanMessage, SystemMessage

from app.agent.nodes import get_llm
from app.config import get_settings
from app.services.embedding_service import encode_single
from app.services.rerank_service import rerank

settings = get_settings()

QA_PATH = os.path.join(os.path.dirname(__file__), "..", "..", "InduOCRBench", "RAG_eval", "QA_pairs.jsonl")
EVAL_CHROMA_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "data", "chroma_eval_ocr")
)
COLLECTION_NAME = "induocrbench_ocr_eval"
CACHE_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "data", "ocr_md_cache")
)

FRIENDLY_CATEGORIES = {
    "Basic Recognition",
    "Structural Alignment",
    "Cross-Field Continuity",
    "Complex Reasoning",
}

ELECTRONIC = {"font", "long", "wide"}

# LLM judge 结果缓存（同一题的判定结果存盘，重跑免再调 API）
JUDGE_CACHE = os.path.join(os.path.dirname(__file__), "judge_cache.jsonl")


def load_friendly_qa(sample_n: int | None = None) -> list[dict]:
    items = [json.loads(l) for l in open(QA_PATH, encoding="utf-8")]
    friendly = [it for it in items if it.get("question_category") in FRIENDLY_CATEGORIES]
    if sample_n and len(friendly) > sample_n:
        random.seed(42)
        friendly = random.sample(friendly, sample_n)
    return friendly


def _load_judge_cache() -> dict:
    """加载已判定的结果缓存 {question_hash: [bool×5]}。"""
    cache = {}
    if os.path.exists(JUDGE_CACHE):
        for line in open(JUDGE_CACHE, encoding="utf-8"):
            try:
                item = json.loads(line)
                cache[item["q"]] = item["verdicts"]
            except Exception:
                pass
    return cache


def _save_judge(question: str, verdicts: list[bool]):
    with open(JUDGE_CACHE, "a", encoding="utf-8") as f:
        f.write(json.dumps({"q": question, "verdicts": verdicts}, ensure_ascii=False) + "\n")


async def llm_judge(question: str, answer: str, chunks: list[str], cache: dict) -> list[bool]:
    """让 LLM 判定每个 chunk 是否包含回答该问题所需信息。

    返回 [bool] × len(chunks)，顺序与 chunks 一致。
    用 question 做缓存 key（同题不重判）。
    """
    key = question
    if key in cache:
        return cache[key]

    # 拼接 chunks，每个标号
    context_block = "\n\n".join(
        f"[Chunk {i+1}]\n{c[:600]}" for i, c in enumerate(chunks)
    )
    messages = [
        SystemMessage(
            content=(
                "你是检索结果相关性评判员。判断每个 Chunk 是否包含回答用户问题所需的关键信息。\n"
                "判定标准：只要 Chunk 里有能支撑标准答案的内容（不要求逐字一致，语义相关即可），就判 yes。\n"
                "注意：OCR 文本可能有别字或格式差异（如表格 HTML 属性不同），只要语义上包含答案信息就算 yes。\n"
                "严格按格式输出，每行一个结果：\n"
                "Chunk 1: yes/no\nChunk 2: yes/no\n... 依次类推，不要任何其他内容。"
            )
        ),
        HumanMessage(
            content=(
                f"用户问题：{question}\n\n"
                f"标准答案：{answer}\n\n"
                f"待判定的检索结果（共 {len(chunks)} 个）：\n{context_block}"
            )
        ),
    ]
    try:
        llm = get_llm()  # 非流式、无 thinking
        resp = await llm.ainvoke(messages)
        text = resp.content.strip()
        # 解析 "Chunk N: yes/no"
        verdicts = []
        for i in range(len(chunks)):
            matched = False
            for line in text.splitlines():
                line_lower = line.lower().strip()
                if f"chunk {i+1}" in line_lower:
                    verdicts.append("yes" in line_lower)
                    matched = True
                    break
            if not matched:
                verdicts.append(False)
        # 补齐长度
        while len(verdicts) < len(chunks):
            verdicts.append(False)
    except Exception:
        verdicts = [False] * len(chunks)

    cache[key] = verdicts
    _save_judge(question, verdicts)
    return verdicts


async def retrieve_top5(col, question: str) -> list[str]:
    vec = await encode_single(question)
    res = col.query(query_embeddings=[vec], n_results=settings.retrieve_top_k)
    docs = res["documents"][0]
    pairs = await rerank(question, docs, top_k=5)
    return [docs[i] for i, _ in pairs]


async def evaluate(col, qa_items: list[dict], judge_cache: dict) -> dict:
    """LLM-as-judge 评测，返回 Hit@1/3/5 和 MRR。"""
    ks = [1, 3, 5]
    hits = {k: 0 for k in ks}
    mrr = 0.0
    n = len(qa_items)
    for idx, item in enumerate(qa_items):
        top5 = await retrieve_top5(col, item["question"])
        verdicts = await llm_judge(item["question"], item["answer"], top5, judge_cache)
        # Hit@k：前 k 个里是否有 yes
        for k in ks:
            if any(verdicts[:k]):
                hits[k] += 1
        # MRR：第一个 yes 的位置
        for r, v in enumerate(verdicts, 1):
            if v:
                mrr += 1.0 / r
                break
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
    parser.add_argument("--sample", type=int, default=50)
    args = parser.parse_args()

    print("=" * 60)
    print("OCR→RAG 端到端评测（LLM-as-judge 判定）")
    print("=" * 60)

    qa_all = [json.loads(l) for l in open(QA_PATH, encoding="utf-8")]
    friendly = [it for it in qa_all if it.get("question_category") in FRIENDLY_CATEGORIES]
    random.seed(42)
    sample = random.sample(friendly, args.sample)
    # 只保留有缓存的
    cached = {f.replace(".md", "") for f in os.listdir(CACHE_DIR) if f.endswith(".md")}
    sample = [q for q in sample if q["filename"].replace(".md", "") in cached]
    print(f"有效题目: {len(sample)} 题\n")

    col = chromadb.PersistentClient(path=EVAL_CHROMA_DIR).get_collection(COLLECTION_NAME)
    judge_cache = _load_judge_cache()
    print(f"judge 缓存: {len(judge_cache)} 题\n")

    # 按文档类型分桶
    groups = {"electronic": [], "scanned": []}
    for item in sample:
        dt = item.get("doc_type", "")
        if dt in ELECTRONIC:
            groups["electronic"].append(item)
        else:
            groups["scanned"].append(item)

    print(f"电子版(font/long/wide): {len(groups['electronic'])} 题")
    print(f"扫描难题: {len(groups['scanned'])} 题\n")

    results = {}
    for gname, items in groups.items():
        if not items:
            continue
        print(f"[{gname}] 评测 {len(items)} 题（LLM-as-judge）...")
        results[gname] = await evaluate(col, items, judge_cache)
        print(f"  结果: {results[gname]}\n")

    # 整体
    all_items = groups["electronic"] + groups["scanned"]
    print(f"[整体] 评测 {len(all_items)} 题...")
    results["overall"] = await evaluate(col, all_items, judge_cache)
    print(f"  结果: {results['overall']}\n")

    print("=" * 60)
    print("分层结果汇总（LLM-as-judge）")
    print("=" * 60)
    print(f"{'文档类型':<12} | {'题数':>4} | {'Hit@1':>6} | {'Hit@3':>6} | {'Hit@5':>6} | {'MRR':>6}")
    print("-" * 56)
    for gname, label in [("electronic", "电子版"), ("scanned", "扫描难题"), ("overall", "整体")]:
        if gname in results:
            r = results[gname]
            print(f"{label:<12} | {r['n']:>4} | {r['Hit@1']:>6} | {r['Hit@3']:>6} | {r['Hit@5']:>6} | {r['MRR']:>6}")
    print(f"\n（对比：标准标注整体 Hit@3 = 0.94）")

    out = os.path.join(os.path.dirname(__file__), "eval_result_ocr_llmjudge.json")
    json.dump(results, open(out, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    print(f"\n结果已保存: {out}")


if __name__ == "__main__":
    asyncio.run(main())
