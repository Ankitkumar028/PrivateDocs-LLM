"""Detailed inspection of the 8 retrieval failure questions."""

import json
from pathlib import Path
import pickle

with open("results/retrieval_metrics.json", "r", encoding="utf-8") as f:
    data = json.load(f)

with open("data/bm25_index.pkl", "rb") as f:
    chunks = pickle.load(f)["chunks"]
chunk_map = {c.chunk_id: c for c in chunks}

with open("eval/datasets/eval_questions_40.jsonl", "r", encoding="utf-8") as f:
    q_map = {q["id"]: q for q in [json.loads(line) for line in f if line.strip()]}

unfiltered = data["new_chunk_level_unfiltered"]
missed = [q for q in unfiltered["details"] if q["matched_rank"] is None or q["matched_rank"] > 10]

print(f"Total missed at Top-10: {len(missed)}")

results = []
for m in missed:
    q_id = m["id"]
    q_obj = q_map[q_id]
    target_cid = q_obj.get("target_chunk_id")
    target_chunk = chunk_map.get(target_cid)
    content = target_chunk.content if target_chunk else ""
    ev = m["evidence_phrase"]

    # Analysis: table row or boundary straddle?
    in_table = "|" in ev or ("\n|" in content and ev in content)
    straddles_boundary = False
    
    # Check if phrase occurs near chunk start/end or was truncated
    idx = content.lower().find(ev.lower())
    if idx != -1:
        if idx < 40 or (idx + len(ev)) > len(content) - 40:
            straddles_boundary = True

    diagnosis = []
    if in_table:
        diagnosis.append("Table Row (dense/BM25 fails to match structured pipe syntax)")
    if straddles_boundary:
        diagnosis.append("Chunk Boundary / Near Boundary Truncation")
    if not diagnosis:
        diagnosis.append("Vocabulary Mismatch / Competing Chunks in Same Document")

    item = {
        "id": q_id,
        "question": m["question"],
        "target_doc": m["target_doc"],
        "evidence_phrase": ev,
        "diagnosis": " + ".join(diagnosis),
        "target_chunk_preview": content[:160].replace("\n", " "),
        "top_3_retrieved": [
            {
                "rank": i + 1,
                "doc_id": ret["doc_id"],
                "section": ret["section"],
                "score": ret["score"],
                "snippet": ret["snippet"].replace("\n", " ")[:140],
            }
            for i, ret in enumerate(m["top_retrievals"][:3])
        ],
    }
    results.append(item)

print(json.dumps(results, indent=2))
