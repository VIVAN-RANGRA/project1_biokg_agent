"""Download STRING protein-protein interactions for human.

Usage:
    python scripts/ingest_string.py --output data/string_ppi.jsonl --score-threshold 700
    python scripts/ingest_string.py --output data/string_ppi_full.jsonl --score-threshold 0
"""

from __future__ import annotations

import argparse
import gzip
import io
import json
import os
import signal
import sys
import time
from pathlib import Path
from typing import Dict

import requests

STRING_LINKS_URL = (
    "https://stringdb-static.org/download/protein.links.v12.0/"
    "9606.protein.links.v12.0.txt.gz"
)
STRING_ALIASES_URL = (
    "https://stringdb-static.org/download/protein.aliases.v12.0/"
    "9606.protein.aliases.v12.0.txt.gz"
)

_shutdown = False


def _handle_signal(signum, frame):
    global _shutdown
    _shutdown = True
    print("\n[!] Ctrl+C detected -- saving partial progress ...")


signal.signal(signal.SIGINT, _handle_signal)


def _download_stream(url: str, description: str, timeout: int = 30,
                      retries: int = 3) -> bytes:
    """Download a URL with retries, returning raw bytes."""
    last_exc = None
    for attempt in range(retries):
        try:
            print(f"  Downloading {description} (attempt {attempt + 1}) ...")
            resp = requests.get(url, timeout=timeout, stream=True)
            resp.raise_for_status()
            chunks = []
            downloaded = 0
            for chunk in resp.iter_content(chunk_size=1024 * 1024):
                chunks.append(chunk)
                downloaded += len(chunk)
                if downloaded % (10 * 1024 * 1024) == 0:
                    print(f"    ... {downloaded / (1024 * 1024):.0f} MB downloaded")
            print(f"    Download complete: {downloaded / (1024 * 1024):.1f} MB")
            return b"".join(chunks)
        except (requests.RequestException, requests.HTTPError) as exc:
            last_exc = exc
            wait = 2 ** attempt
            print(f"  [retry {attempt + 1}/{retries}] {exc} -- waiting {wait}s")
            time.sleep(wait)
    raise RuntimeError(f"Failed to download {description}: {last_exc}")


def _looks_like_gene_symbol(alias: str) -> bool:
    """Heuristic for HGNC-like symbols (e.g., TP53, BRCA1, MAP2K1)."""
    return bool(alias) and alias.upper() == alias and 2 <= len(alias) <= 15


def _pick_alias(current: str | None, alias: str, source: str, preferred_sources: set[str]) -> str:
    """Pick the best alias for display/storage."""
    if not current:
        return alias
    # Highest priority: preferred STRING/HGNC mapping sources.
    if source in preferred_sources:
        return alias
    # Otherwise prefer HGNC-like symbols over generic aliases.
    if _looks_like_gene_symbol(alias) and not _looks_like_gene_symbol(current):
        return alias
    return current


def build_alias_map(aliases_gz: bytes) -> Dict[str, str]:
    """Build mapping from STRING protein ID -> gene symbol.

    We prefer aliases of source 'Ensembl_HGNC' or 'BioMart_HUGO' as gene symbols.
    Falls back to the first alias seen.
    """
    print("[aliases] Parsing STRING aliases ...")
    protein_to_gene: Dict[str, str] = {}
    preferred_sources = {"Ensembl_HGNC", "BioMart_HUGO", "Ensembl_HGNC_symbol",
                         "Ensembl_HGNC_UniProt_ID_(mapping_supplied_by_UniProt)",
                         "BLAST_UniProt_GN"}
    line_count = 0

    with gzip.open(io.BytesIO(aliases_gz), "rt", encoding="utf-8") as fh:
        # skip header
        header = next(fh, None)
        for line in fh:
            line_count += 1
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 3:
                continue
            string_id, alias, source = parts[0], parts[1], parts[2]

            protein_to_gene[string_id] = _pick_alias(
                protein_to_gene.get(string_id),
                alias,
                source,
                preferred_sources,
            )

            if line_count % 1_000_000 == 0:
                print(f"  [aliases] Processed {line_count:,} lines")

    print(f"  [aliases] Done: {len(protein_to_gene):,} protein-to-gene mappings from {line_count:,} lines")
    return protein_to_gene


def main() -> None:
    parser = argparse.ArgumentParser(description="Download STRING PPI data for human.")
    parser.add_argument("--output", type=str, default="data/string_ppi.jsonl",
                        help="Output JSONL path")
    parser.add_argument("--score-threshold", type=int, default=700,
                        help="Minimum combined score (0-1000)")
    parser.add_argument(
        "--keep-unmapped",
        action="store_true",
        help="Keep interactions even when alias mapping is missing (use STRING protein IDs as fallback).",
    )
    parser.add_argument("--aliases-cache", type=str, default="data/_string_aliases.gz",
                        help="Cache path for aliases gzip")
    parser.add_argument("--links-cache", type=str, default="data/_string_links.gz",
                        help="Cache path for links gzip")
    args = parser.parse_args()

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # --- Step 1: Download / cache aliases ---
    aliases_cache = Path(args.aliases_cache)
    if aliases_cache.exists():
        print(f"[cache] Using cached aliases: {aliases_cache}")
        aliases_gz = aliases_cache.read_bytes()
    else:
        aliases_gz = _download_stream(STRING_ALIASES_URL, "STRING aliases")
        aliases_cache.parent.mkdir(parents=True, exist_ok=True)
        aliases_cache.write_bytes(aliases_gz)
        print(f"[cache] Saved aliases cache: {aliases_cache}")

    protein_to_gene = build_alias_map(aliases_gz)
    # Free alias bytes
    del aliases_gz

    # --- Step 2: Download / cache links ---
    links_cache = Path(args.links_cache)
    if links_cache.exists():
        print(f"[cache] Using cached links: {links_cache}")
        links_gz = links_cache.read_bytes()
    else:
        links_gz = _download_stream(STRING_LINKS_URL, "STRING links")
        links_cache.parent.mkdir(parents=True, exist_ok=True)
        links_cache.write_bytes(links_gz)
        print(f"[cache] Saved links cache: {links_cache}")

    # --- Step 3: Stream-parse links and write JSONL ---
    print(f"\n[parse] Filtering interactions with score >= {args.score_threshold} ...")
    written = 0
    skipped = 0
    line_count = 0

    with gzip.open(io.BytesIO(links_gz), "rt", encoding="utf-8") as fh, \
         open(output_path, "w", encoding="utf-8") as out_fh:

        header = next(fh, None)  # skip header: protein1 protein2 combined_score
        for line in fh:
            if _shutdown:
                break
            line_count += 1
            parts = line.strip().split()
            if len(parts) < 3:
                continue

            protein1, protein2 = parts[0], parts[1]
            try:
                score = int(parts[2])
            except ValueError:
                continue

            if score < args.score_threshold:
                skipped += 1
                if line_count % 1_000_000 == 0:
                    print(f"  Processed {line_count:,} lines | kept {written:,} | skipped {skipped:,}")
                continue

            gene1 = protein_to_gene.get(protein1)
            gene2 = protein_to_gene.get(protein2)
            if args.keep_unmapped:
                gene1 = gene1 or protein1
                gene2 = gene2 or protein2
            if not gene1 or not gene2:
                skipped += 1
                continue

            record = {"gene": gene1, "partner": gene2, "score": score}
            out_fh.write(json.dumps(record, ensure_ascii=False) + "\n")
            written += 1

            if line_count % 1_000_000 == 0:
                print(f"  Processed {line_count:,} lines | kept {written:,} | skipped {skipped:,}")

    # Free link bytes
    del links_gz

    print(f"\n=== Done! {written:,} interactions written (from {line_count:,} lines) ===")
    print(f"Output: {output_path.resolve()}")


if __name__ == "__main__":
    main()
