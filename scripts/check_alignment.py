import json
from pathlib import Path

with open("eval/datasets/eval_questions_40.jsonl", "r", encoding="utf-8") as f:
    questions = [json.loads(line) for line in f if line.strip()]

checked = {
    "q_meta_2", "q_amd_4", "q_amd_2", "q_amd_3",
    "q_nvda_2", "q_msft_2", "q_amzn_4", "q_aapl_2", "q_goog_2"
}

print(f"{'QID':<12} | {'Question':<55} | {'Evidence Phrase'}")
print("-" * 105)
for q in questions:
    if q["id"] in checked or q.get("type") == "unanswerable":
        continue
    q_txt = q["question"][:52] + "..." if len(q["question"]) > 52 else q["question"]
    print(f"{q['id']:<12} | {q_txt:<55} | '{q['evidence_phrase']}'")
