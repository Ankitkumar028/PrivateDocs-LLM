# Evaluation Suite

This directory contains evaluation datasets, evaluation harnesses, and reporting tools.

## Structure
- `datasets/corpus_manifest.yaml`: URLs and hashes for public SEC 10-K filings and policy documents. Large binary files are never committed directly.
- `datasets/sample_questions.jsonl`: Questions with reference answers, source doc/page, and question type (factual, numeric, multi-chunk, unanswerable).
- `build_corpus.py`: Downloads documents listed in `corpus_manifest.yaml` adhering to fair-access guidelines.
- `generate_questions.py`: LLM-assisted question generator for human curation.
- `run_retrieval_eval.py`: Computes recall@k (k=3, 5, 10) and MRR across chunking and retrieval modes.
- `run_answer_eval.py`: Evaluates correctness, faithfulness, and abstention rate on unanswerable questions.
- `judge.py`: Fixed-rubric LLM-as-a-judge implementation.
- `report.py`: Aggregates evaluation runs into tables and charts.
