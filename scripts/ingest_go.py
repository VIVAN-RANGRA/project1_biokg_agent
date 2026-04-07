"""Download GO terms and gene-GO annotations.

Usage:
    python scripts/ingest_go.py --output-terms data/go_terms.jsonl --output-annotations data/gene_annotations.jsonl
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
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Set

import requests

GO_OBO_URL = "https://current.geneontology.org/ontology/go-basic.obo"
GOA_HUMAN_URL = "https://current.geneontology.org/annotations/goa_human.gaf.gz"

_shutdown = False


def _handle_signal(signum, frame):
    global _shutdown
    _shutdown = True
    print("\n[!] Ctrl+C detected -- saving partial progress ...")


signal.signal(signal.SIGINT, _handle_signal)


def _download_with_retry(url: str, description: str, timeout: int = 60,
                          retries: int = 3) -> bytes:
    """Download URL content with retries and exponential back-off."""
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
                if downloaded % (5 * 1024 * 1024) == 0:
                    print(f"    ... {downloaded / (1024 * 1024):.0f} MB")
            print(f"    Download complete: {downloaded / (1024 * 1024):.1f} MB")
            return b"".join(chunks)
        except (requests.RequestException, requests.HTTPError) as exc:
            last_exc = exc
            wait = 2 ** attempt
            print(f"  [retry {attempt + 1}/{retries}] {exc} -- waiting {wait}s")
            time.sleep(wait)
    raise RuntimeError(f"Failed to download {description}: {last_exc}")


# ---------------------------------------------------------------------------
# OBO parser (simple text-based, no library needed)
# ---------------------------------------------------------------------------
def parse_obo(obo_text: str) -> List[dict]:
    """Parse GO OBO format and return list of term dicts."""
    terms: List[dict] = []
    current_term: dict | None = None

    for line in obo_text.splitlines():
        line = line.strip()

        if line == "[Term]":
            if current_term and "id" in current_term:
                terms.append(current_term)
            current_term = {}
            continue

        if line.startswith("[") and line.endswith("]"):
            # Other stanza (e.g., [Typedef]) -- save previous term if any
            if current_term and "id" in current_term:
                terms.append(current_term)
            current_term = None
            continue

        if current_term is None:
            continue

        if line.startswith("id: "):
            current_term["id"] = line[4:].strip()
        elif line.startswith("name: "):
            current_term["name"] = line[6:].strip()
        elif line.startswith("namespace: "):
            current_term["namespace"] = line[11:].strip()
        elif line.startswith("def: "):
            # Definition is quoted: def: "some text" [source]
            defn = line[5:].strip()
            if defn.startswith('"'):
                end_quote = defn.find('"', 1)
                if end_quote > 0:
                    defn = defn[1:end_quote]
            current_term["definition"] = defn
        elif line.startswith("is_obsolete: true"):
            current_term["obsolete"] = True

    # Last term
    if current_term and "id" in current_term:
        terms.append(current_term)

    # Filter out obsolete terms
    terms = [t for t in terms if not t.get("obsolete")]
    return terms


# ---------------------------------------------------------------------------
# GAF parser (Gene Association File)
# ---------------------------------------------------------------------------
def parse_gaf_stream(gaf_gz: bytes) -> Dict[str, Set[str]]:
    """Parse GAF (gzip), returning gene_symbol -> set of GO IDs.

    GAF 2.x columns (tab-separated):
      0: DB
      1: DB Object ID
      2: DB Object Symbol  <-- gene symbol
      3: Qualifier
      4: GO ID             <-- GO term
      5-16: other fields
    """
    gene_to_go: Dict[str, Set[str]] = defaultdict(set)
    line_count = 0

    with gzip.open(io.BytesIO(gaf_gz), "rt", encoding="utf-8") as fh:
        for line in fh:
            if _shutdown:
                break
            if line.startswith("!"):
                continue
            line_count += 1
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 5:
                continue

            gene_symbol = parts[2].strip()
            go_id = parts[4].strip()

            if gene_symbol and go_id.startswith("GO:"):
                gene_to_go[gene_symbol].add(go_id)

            if line_count % 500_000 == 0:
                print(f"  [gaf] Processed {line_count:,} lines | {len(gene_to_go):,} genes")

    print(f"  [gaf] Done: {line_count:,} lines | {len(gene_to_go):,} genes with annotations")
    return gene_to_go


def main() -> None:
    parser = argparse.ArgumentParser(description="Download GO terms and gene annotations.")
    parser.add_argument("--output-terms", type=str, default="data/go_terms.jsonl",
                        help="Output JSONL for GO terms")
    parser.add_argument("--output-annotations", type=str, default="data/gene_annotations.jsonl",
                        help="Output JSONL for gene-GO annotations")
    parser.add_argument("--obo-cache", type=str, default="data/_go_basic.obo",
                        help="Cache path for OBO file")
    parser.add_argument("--gaf-cache", type=str, default="data/_goa_human.gaf.gz",
                        help="Cache path for GAF gzip")
    args = parser.parse_args()

    terms_path = Path(args.output_terms)
    annot_path = Path(args.output_annotations)
    terms_path.parent.mkdir(parents=True, exist_ok=True)
    annot_path.parent.mkdir(parents=True, exist_ok=True)

    # --- Step 1: GO OBO terms ---
    obo_cache = Path(args.obo_cache)
    if obo_cache.exists():
        print(f"[cache] Using cached OBO: {obo_cache}")
        obo_text = obo_cache.read_text(encoding="utf-8")
    else:
        obo_bytes = _download_with_retry(GO_OBO_URL, "GO OBO file")
        obo_text = obo_bytes.decode("utf-8")
        obo_cache.parent.mkdir(parents=True, exist_ok=True)
        obo_cache.write_text(obo_text, encoding="utf-8")
        print(f"[cache] Saved OBO cache: {obo_cache}")
        del obo_bytes

    print("[obo] Parsing GO terms ...")
    terms = parse_obo(obo_text)
    del obo_text  # free memory

    print(f"[obo] Parsed {len(terms):,} non-obsolete GO terms")

    with open(terms_path, "w", encoding="utf-8") as out_fh:
        for term in terms:
            record = {
                "id": term.get("id", ""),
                "name": term.get("name", ""),
                "namespace": term.get("namespace", ""),
            }
            if "definition" in term:
                record["definition"] = term["definition"]
            out_fh.write(json.dumps(record, ensure_ascii=False) + "\n")

    print(f"[obo] Wrote {len(terms):,} GO terms to {terms_path}")
    del terms

    if _shutdown:
        print("[!] Interrupted -- GO terms saved, skipping annotations")
        return

    # --- Step 2: Gene-GO annotations ---
    gaf_cache = Path(args.gaf_cache)
    if gaf_cache.exists():
        print(f"[cache] Using cached GAF: {gaf_cache}")
        gaf_gz = gaf_cache.read_bytes()
    else:
        gaf_gz = _download_with_retry(GOA_HUMAN_URL, "GOA human annotations")
        gaf_cache.parent.mkdir(parents=True, exist_ok=True)
        gaf_cache.write_bytes(gaf_gz)
        print(f"[cache] Saved GAF cache: {gaf_cache}")

    print("[gaf] Parsing gene annotations ...")
    gene_to_go = parse_gaf_stream(gaf_gz)
    del gaf_gz

    written = 0
    with open(annot_path, "w", encoding="utf-8") as out_fh:
        for gene, go_ids in sorted(gene_to_go.items()):
            record = {"gene": gene, "go_ids": sorted(go_ids)}
            out_fh.write(json.dumps(record, ensure_ascii=False) + "\n")
            written += 1

    print(f"\n=== Done! ===")
    print(f"  GO terms:      {terms_path.resolve()}")
    print(f"  Annotations:   {annot_path.resolve()} ({written:,} genes)")


if __name__ == "__main__":
    main()
