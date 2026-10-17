"""Script to download public documents listed in eval/datasets/corpus_manifest.yaml."""

from __future__ import annotations

import argparse
from pathlib import Path

import httpx
import yaml


def download_corpus(manifest_path: Path, output_dir: Path) -> None:
    """Download public documents respecting fair access guidelines."""
    with open(manifest_path, encoding="utf-8") as f:
        data = yaml.safe_load(f)

    docs = data.get("documents", [])
    output_dir.mkdir(parents=True, exist_ok=True)

    headers = {
        "User-Agent": "PrivateDocs-ResearchBot/1.0 (academic-eval; contact@privatedocs.internal)"
    }

    print(f"Loaded manifest with {len(docs)} documents.")
    for doc in docs:
        doc_id = doc["id"]
        url = doc["url"]
        fmt = doc.get("format", "txt")
        target_file = output_dir / f"{doc_id}.{fmt}"

        if target_file.exists():
            print(f"[Skip] {doc_id} already exists at {target_file}")
            continue

        print(f"[Downloading] {doc_id} from {url}...")
        try:
            with httpx.Client(headers=headers, follow_redirects=True, timeout=30.0) as client:
                resp = client.get(url)
                if resp.status_code == 200:
                    target_file.write_bytes(resp.content)
                    print(f"[Done] Saved {doc_id} ({len(resp.content)} bytes)")
                else:
                    print(f"[Warning] Failed to fetch {url}: HTTP {resp.status_code}")
        except Exception as exc:
            print(f"[Error] Failed downloading {doc_id}: {exc}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Download corpus documents from manifest.")
    parser.add_argument("--manifest", default="eval/datasets/corpus_manifest.yaml", help="Manifest path")
    parser.add_argument("--out", default="data/raw_docs", help="Output directory")
    args = parser.parse_args()

    download_corpus(Path(args.manifest), Path(args.out))


if __name__ == "__main__":
    main()
