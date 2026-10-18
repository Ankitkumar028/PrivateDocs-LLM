import json
from pathlib import Path

path = Path("eval/datasets/eval_questions_40.jsonl")
with open(path, "r", encoding="utf-8") as f:
    questions = [json.loads(line) for line in f if line.strip()]

updates = {
    "q_aapl_2": {
        "evidence_phrase": "Total undiscounted liabilities | 13,841 | 1,425 | 15,266",
        "reference_answer": "In Note 7 (Leases), Apple reported total undiscounted lease liabilities of $15,266 million ($13,841 million operating, $1,425 million manufacturing), with imputed interest of $(2,424) million.",
    },
    "q_msft_2": {
        "evidence_phrase": "was $ 229 billion as of June 30, 2023, of which $ 224",
        "reference_answer": "As of June 30, 2023, Microsoft reported commercial remaining performance obligation of $224 billion (out of $229 billion total remaining performance obligations).",
    },
    "q_nvda_2": {
        "evidence_phrase": "Data Center | $ | 47,525 | $ | 15,005",
        "reference_answer": "NVIDIA's Data Center revenue was $47,525 million in fiscal year 2024, compared to $15,005 million in fiscal year 2023 (an increase of 217%).",
    },
    "q_goog_2": {
        "evidence_phrase": "Class C share repurchases | 469 | 52,577",
        "reference_answer": "Alphabet spent $52,577 million to repurchase 469 million shares of Class C capital stock in 2023.",
    },
    "q_amzn_3": {
        "question": "How did Amazon describe the business offerings and customer scope of its AWS segment?",
        "evidence_phrase": "The AWS segment consists of amounts earned from global sales of compute, storage, database, and other services",
        "reference_answer": "Amazon stated that the AWS segment consists of amounts earned from global sales of compute, storage, database, and other services for start-ups, enterprises, government agencies, and academic institutions.",
    },
    "q_amzn_4": {
        "evidence_phrase": "Unearned revenue is recorded when payments are received or due in advance of performing our service obligations and is recognized over the service period.",
        "reference_answer": "Amazon records unearned revenue when payments are received or due in advance of performing service obligations, and recognizes it over the service period as services are provided.",
    },
    "q_amd_2": {
        "evidence_phrase": "Acquisition-related intangible amortization expense was $ 2.8 billion and $ 3.5 billion in fiscal year 2023 and 2022, respectively.",
        "reference_answer": "In Note 6, AMD disclosed that acquisition-related intangible amortization expense (primarily from the Xilinx acquisition) was $2.8 billion in 2023 compared to $3.5 billion in 2022.",
    },
    "q_amd_3": {
        "evidence_phrase": "Data Center | $ | 6,496 | $ | 6,043",
        "reference_answer": "AMD reported Data Center segment net revenue of $6,496 million in 2023 compared to $6,043 million in 2022 (an increase of 7%).",
    },
    "q_amd_4": {
        "evidence_phrase": "25% decrease in Client segment revenue primarily due to lower processor sales and a 9% decrease in Gaming segment revenue",
        "reference_answer": "In 2023, AMD's Client segment revenue decreased 25% primarily due to lower processor sales in the PC market, and Gaming segment revenue decreased 9% primarily due to lower semi-custom product sales.",
    },
    "q_meta_2": {
        "evidence_phrase": "Ad impressions delivered across our Family of Apps increased 28% year-over-year in 2023, partially offset by a 9% year-over-year decrease in the average price per ad.",
        "reference_answer": "In 2023, ad impressions delivered across Meta's Family of Apps increased 28% year-over-year, while the average price per ad decreased 9% year-over-year.",
    },
}

for q in questions:
    qid = q["id"]
    if qid in updates:
        for k, v in updates[qid].items():
            q[k] = v
        print(f"Updated {qid}")

with open(path, "w", encoding="utf-8") as f:
    for q in questions:
        f.write(json.dumps(q) + "\n")

print(f"Successfully wrote {len(questions)} updated questions to {path}")
