"""Corpus downloader for SEC EDGAR 10-K filings and public policy PDFs.

Complies with SEC Fair Access Policy:
- Explicit descriptive User-Agent: "<App>/<Ver> (<Contact Email>; <Affiliation>)"
- Strict rate limiting: <= 4 requests/sec (min 0.25s delay between queries)
- Exponential backoff on HTTP 429 / 503
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path
from typing import Any

import httpx
import yaml

# Force UTF-8 stdout if supported
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("privatedocs.eval.build_corpus")

try:
    from privatedocs.config import get_settings
    SEC_USER_AGENT = get_settings().SEC_USER_AGENT
except Exception:
    SEC_USER_AGENT = "PrivateDocs-LLM-Evaluation/1.0 (admin@example.com; Research-RAG-Evaluation)"

REQUEST_DELAY_SECONDS = 0.3  # Max ~3.3 req/s, well below the 10 req/s SEC limit


def download_file_with_retry(
    client: httpx.Client,
    url: str,
    target_path: Path,
    max_retries: int = 3,
) -> bool:
    """Download single file with rate limiting and exponential backoff on rate-limits."""
    for attempt in range(1, max_retries + 1):
        try:
            time.sleep(REQUEST_DELAY_SECONDS)
            resp = client.get(url)

            if resp.status_code == 200:
                target_path.write_bytes(resp.content)
                size_kb = len(resp.content) / 1024.0
                logger.info(
                    "Downloaded %s (%.1f KB) -> %s",
                    target_path.name,
                    size_kb,
                    target_path,
                )
                return True

            if resp.status_code == 429:
                retry_after = float(resp.headers.get("Retry-After", 2.0 * attempt))
                logger.warning(
                    "Rate limit hit (429) for %s. Backing off for %.1fs (attempt %d/%d)...",
                    url,
                    retry_after,
                    attempt,
                    max_retries,
                )
                time.sleep(retry_after)
                continue

            logger.error("Failed to download %s: HTTP %d - %s", url, resp.status_code, resp.text[:120])
            if resp.status_code >= 500:
                time.sleep(1.0 * attempt)
                continue
            return False

        except Exception as exc:
            logger.warning("Download error for %s (%s). Retrying in %.1fs...", url, exc, 1.0 * attempt)
            time.sleep(1.0 * attempt)

    return False


def build_corpus(
    manifest_path: Path,
    output_dir: Path,
    max_docs: int | None = None,
) -> dict[str, Any]:
    """Download manifest documents into target directory."""
    if not manifest_path.exists():
        raise FileNotFoundError(f"Manifest not found at {manifest_path}")

    with open(manifest_path, encoding="utf-8") as f:
        data = yaml.safe_load(f)

    docs = data.get("documents", [])
    if max_docs:
        docs = docs[:max_docs]

    output_dir.mkdir(parents=True, exist_ok=True)
    logger.info("Starting corpus download for %d documents -> %s", len(docs), output_dir)

    headers = {
        "User-Agent": SEC_USER_AGENT,
        "Accept": "*/*",
        "Accept-Encoding": "gzip, deflate",
    }

    results: dict[str, Any] = {
        "total": len(docs),
        "downloaded": 0,
        "skipped": 0,
        "failed": 0,
        "files": [],
    }

    t0 = time.perf_counter()

    with httpx.Client(headers=headers, follow_redirects=True, timeout=45.0) as client:
        for doc in docs:
            doc_id = doc["id"]
            url = doc["url"]
            fmt = doc.get("format", "html")
            target_file = output_dir / f"{doc_id}.{fmt}"

            if target_file.exists() and target_file.stat().st_size > 0:
                logger.info("File already exists, skipping: %s", target_file.name)
                results["skipped"] += 1
                results["files"].append(
                    {
                        "id": doc_id,
                        "path": str(target_file),
                        "size_bytes": target_file.stat().st_size,
                        "status": "cached",
                    }
                )
                continue

            success = download_file_with_retry(client, url, target_file)
            if success:
                results["downloaded"] += 1
                results["files"].append(
                    {
                        "id": doc_id,
                        "path": str(target_file),
                        "size_bytes": target_file.stat().st_size,
                        "status": "downloaded",
                    }
                )
            else:
                results["failed"] += 1
                results["files"].append(
                    {"id": doc_id, "url": url, "status": "failed"}
                )

    results["duration_sec"] = round(time.perf_counter() - t0, 2)
    logger.info(
        "Corpus build finished in %.1fs: %d downloaded, %d skipped, %d failed.",
        results["duration_sec"],
        results["downloaded"],
        results["skipped"],
        results["failed"],
    )
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="Download evaluation corpus from manifest.")
    parser.add_argument(
        "--manifest",
        default="eval/datasets/corpus_manifest.yaml",
        help="Path to manifest yaml",
    )
    parser.add_argument(
        "--out",
        default="data/raw_docs",
        help="Destination directory for raw documents",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional maximum number of documents to download",
    )
    args = parser.parse_args()

    build_corpus(
        manifest_path=Path(args.manifest),
        output_dir=Path(args.out),
        max_docs=args.limit,
    )


if __name__ == "__main__":
    main()
