"""Download PubMed abstracts in batches. Memory-safe.

Usage:
    python scripts/ingest_pubmed.py --genes 50 --abstracts-per-gene 200 --output data/pubmed_abstracts.jsonl --batch-size 100

This will download ~10,000 abstracts for the top genes in this script.
For large-scale runs, pass a custom list:
    python scripts/ingest_pubmed.py --gene-list data/top_500_genes.txt --genes 500 --abstracts-per-gene 1200
"""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import json
import os
import signal
import sys
import time
from pathlib import Path
from typing import List, Set
from xml.etree import ElementTree

import requests

# ---------------------------------------------------------------------------
# Baseline seed genes (can be overridden by --gene-list)
# ---------------------------------------------------------------------------
TOP_GENES: List[str] = [
    "TP53", "BRCA1", "EGFR", "KRAS", "MYC", "PTEN", "AKT1", "BRAF",
    "PIK3CA", "RB1", "ERBB2", "ALK", "VEGFA", "MTOR", "CDK4", "ATM",
    "CHEK2", "PALB2", "RAD51", "MDM2", "JAK2", "FLT3", "ABL1", "BCL2",
    "NOTCH1", "CTNNB1", "APC", "VHL", "IDH1", "IDH2", "NRAS", "MAP2K1",
    "RAF1", "SRC", "FGFR1", "FGFR2", "FGFR3", "MET", "RET", "ROS1",
    "KIT", "PDGFRA", "SMAD4", "STK11", "NF1", "NF2", "TSC1", "TSC2",
    "WT1", "PTCH1",
]

# NCBI Entrez base URLs
ESEARCH_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
EFETCH_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"

# Graceful shutdown flag
_shutdown = False


def _handle_signal(signum, frame):
    global _shutdown
    _shutdown = True
    print("\n[!] Ctrl+C detected -- finishing current batch then saving partial progress ...")


signal.signal(signal.SIGINT, _handle_signal)


def _rate_delay(has_key: bool) -> float:
    return 0.1 if has_key else 0.34


def _request_with_retry(url: str, params: dict, timeout: int = 30, retries: int = 3,
                         delay: float = 0.34) -> requests.Response:
    """HTTP GET with exponential back-off retries."""
    last_exc = None
    for attempt in range(retries):
        try:
            resp = requests.get(url, params=params, timeout=timeout)
            resp.raise_for_status()
            return resp
        except (requests.RequestException, requests.HTTPError) as exc:
            last_exc = exc
            wait = delay * (2 ** attempt)
            print(f"  [retry {attempt + 1}/{retries}] {exc} -- waiting {wait:.1f}s")
            time.sleep(wait)
    raise RuntimeError(f"Failed after {retries} retries: {last_exc}")


def esearch(gene: str, retmax: int, api_key: str | None, email: str | None) -> List[str]:
    """Return list of PubMed IDs for a gene search query."""
    params = {
        "db": "pubmed",
        "term": f"{gene}[Gene Name] AND Homo sapiens[Organism]",
        "retmax": retmax,
        "retmode": "json",
        "sort": "relevance",
    }
    if api_key:
        params["api_key"] = api_key
    if email:
        params["email"] = email

    resp = _request_with_retry(ESEARCH_URL, params, delay=_rate_delay(bool(api_key)))
    data = resp.json()
    return data.get("esearchresult", {}).get("idlist", [])


def efetch_batch(pmids: List[str], api_key: str | None, email: str | None) -> List[dict]:
    """Fetch article metadata for a batch of PMIDs. Returns list of dicts."""
    params = {
        "db": "pubmed",
        "id": ",".join(pmids),
        "rettype": "xml",
        "retmode": "xml",
    }
    if api_key:
        params["api_key"] = api_key
    if email:
        params["email"] = email

    resp = _request_with_retry(EFETCH_URL, params, delay=_rate_delay(bool(api_key)))

    results: List[dict] = []
    try:
        root = ElementTree.fromstring(resp.content)
    except ElementTree.ParseError:
        return results

    for article in root.iter("PubmedArticle"):
        pmid_el = article.find(".//PMID")
        title_el = article.find(".//ArticleTitle")
        abstract_el = article.find(".//AbstractText")
        pmid = pmid_el.text.strip() if pmid_el is not None and pmid_el.text else ""
        title = title_el.text.strip() if title_el is not None and title_el.text else ""
        abstract = abstract_el.text.strip() if abstract_el is not None and abstract_el.text else ""
        if pmid and (title or abstract):
            results.append({"pmid": pmid, "title": title, "abstract": abstract})
    return results


def load_existing_pmids(output_path: Path) -> Set[str]:
    """Load already-downloaded PMIDs from an existing JSONL file for resume support."""
    seen: Set[str] = set()
    if not output_path.exists():
        return seen
    with open(output_path, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
                seen.add(rec.get("pmid", ""))
            except json.JSONDecodeError:
                continue
    print(f"[resume] Found {len(seen)} existing PMIDs in {output_path}")
    return seen


def load_gene_list(path: Path) -> List[str]:
    """Load one gene symbol per line, ignoring blanks/comments."""
    genes: List[str] = []
    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            gene = line.strip()
            if not gene or gene.startswith("#"):
                continue
            genes.append(gene.upper())
    return genes


def fetch_gene_records(
    gene: str,
    abstracts_per_gene: int,
    batch_size: int,
    api_key: str | None,
    email: str | None,
    delay: float,
) -> tuple[str, list[dict]]:
    """Fetch all records for one gene (worker-safe, no shared state)."""
    records_out: list[dict] = []
    try:
        pmids = esearch(gene, retmax=abstracts_per_gene, api_key=api_key, email=email)
    except RuntimeError:
        return gene, records_out

    if delay > 0:
        time.sleep(delay)

    if not pmids:
        return gene, records_out

    for batch_start in range(0, len(pmids), batch_size):
        if _shutdown:
            break
        batch = pmids[batch_start: batch_start + batch_size]
        try:
            records = efetch_batch(batch, api_key=api_key, email=email)
        except RuntimeError:
            continue

        for rec in records:
            rec["gene"] = gene
            records_out.append(rec)

        if delay > 0:
            time.sleep(delay)

    return gene, records_out


def main() -> None:
    parser = argparse.ArgumentParser(description="Download PubMed abstracts for target genes.")
    parser.add_argument("--genes", type=int, default=50, help="Number of genes to query from the selected gene list")
    parser.add_argument("--abstracts-per-gene", type=int, default=200, help="Max abstracts per gene")
    parser.add_argument("--output", type=str, default="data/pubmed_abstracts.jsonl", help="Output JSONL path")
    parser.add_argument("--batch-size", type=int, default=100, help="EFetch batch size")
    parser.add_argument("--gene-list", type=str, default=None, help="Optional path to text file with one gene symbol per line")
    parser.add_argument("--workers", type=int, default=1, help="Parallel worker processes (max 4)")
    parser.add_argument(
        "--global-rps",
        type=float,
        default=0.0,
        help="Target total requests/sec across all workers (0 = auto safe default)",
    )
    args = parser.parse_args()

    api_key = os.environ.get("NCBI_API_KEY")
    email = os.environ.get("NCBI_EMAIL")
    has_key = bool(api_key)
    delay = _rate_delay(has_key)
    workers = max(1, min(4, int(args.workers)))

    if has_key:
        print("[config] NCBI_API_KEY detected -- rate limit: 10 req/s")
    else:
        print("[config] No NCBI_API_KEY -- rate limit: 3 req/s  (set NCBI_API_KEY for 10/s)")

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if args.gene_list:
        gene_path = Path(args.gene_list)
        if not gene_path.exists():
            print(f"[error] Gene list not found: {gene_path}")
            sys.exit(1)
        source_genes = load_gene_list(gene_path)
        if not source_genes:
            print(f"[error] Gene list is empty: {gene_path}")
            sys.exit(1)
        print(f"[config] Loaded {len(source_genes)} genes from {gene_path}")
    else:
        source_genes = TOP_GENES
        print(f"[config] Using built-in seed list ({len(source_genes)} genes)")

    genes = source_genes[: args.genes]
    if args.global_rps > 0:
        target_rps = max(0.1, float(args.global_rps))
    else:
        # Keep a conservative total rate to avoid API throttling.
        target_rps = 9.0 if has_key else 2.8
    # Effective per-request delay per worker so aggregate RPS stays bounded.
    delay = max(delay, workers / target_rps)
    print(f"[config] workers={workers}, target_total_rps={target_rps:.2f}, per_worker_delay={delay:.3f}s")
    existing_pmids = load_existing_pmids(output_path)
    total_downloaded = len(existing_pmids)

    # Open in append mode for resume support.
    with open(output_path, "a", encoding="utf-8") as out_fh:
        if workers == 1:
            for gene_idx, gene in enumerate(genes):
                if _shutdown:
                    break
                print(f"\n[{gene_idx + 1}/{len(genes)}] Fetching PubMed for gene: {gene}")
                _, records = fetch_gene_records(
                    gene=gene,
                    abstracts_per_gene=args.abstracts_per_gene,
                    batch_size=args.batch_size,
                    api_key=api_key,
                    email=email,
                    delay=delay,
                )
                gene_new = 0
                for rec in records:
                    pmid = rec.get("pmid", "")
                    if not pmid or pmid in existing_pmids:
                        continue
                    out_fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
                    existing_pmids.add(pmid)
                    gene_new += 1
                    total_downloaded += 1
                out_fh.flush()
                print(f"  Gene {gene}: {gene_new} new records | total: {total_downloaded}")
        else:
            print(f"[parallel] Starting ProcessPoolExecutor with {workers} workers")
            with ProcessPoolExecutor(max_workers=workers) as ex:
                futures = {
                    ex.submit(
                        fetch_gene_records,
                        gene,
                        args.abstracts_per_gene,
                        args.batch_size,
                        api_key,
                        email,
                        delay,
                    ): gene
                    for gene in genes
                }
                completed = 0
                for fut in as_completed(futures):
                    if _shutdown:
                        break
                    gene = futures[fut]
                    completed += 1
                    try:
                        _, records = fut.result()
                    except Exception as exc:  # noqa: BLE001
                        print(f"  [error] {gene}: {exc}")
                        continue

                    gene_new = 0
                    for rec in records:
                        pmid = rec.get("pmid", "")
                        if not pmid or pmid in existing_pmids:
                            continue
                        out_fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
                        existing_pmids.add(pmid)
                        gene_new += 1
                        total_downloaded += 1
                    out_fh.flush()
                    print(
                        f"[{completed}/{len(genes)}] {gene}: {gene_new} new records | "
                        f"total: {total_downloaded}"
                    )

    print(f"\n=== Done! Total records: {total_downloaded} ===")
    print(f"Output: {output_path.resolve()}")


if __name__ == "__main__":
    main()
