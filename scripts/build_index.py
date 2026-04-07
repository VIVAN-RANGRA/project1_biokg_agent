"""Build FAISS index and unified demo_bundle.json from downloaded data.

Usage:
    python scripts/build_index.py --data-dir data/ --output-dir checkpoints/ --batch-size 512
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import re
import signal
import sys
import time
import traceback
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Set

_shutdown = False


def _handle_signal(signum, frame):
    global _shutdown
    _shutdown = True
    print("\n[!] Ctrl+C detected -- saving partial progress ...")


signal.signal(signal.SIGINT, _handle_signal)


def _peak_memory_mb() -> float:
    """Return peak memory usage in MB (best effort, cross-platform)."""
    try:
        import resource
        # Unix
        usage = resource.getrusage(resource.RUSAGE_SELF)
        return usage.ru_maxrss / 1024  # ru_maxrss is in KB on Linux
    except ImportError:
        pass
    try:
        import psutil
        process = psutil.Process(os.getpid())
        return process.memory_info().peak_wss / (1024 * 1024)
    except Exception:
        pass
    return -1.0


def load_jsonl(path: Path) -> List[dict]:
    """Load a JSONL file, returning list of dicts."""
    records = []
    if not path.exists():
        return records
    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    return records


def stream_jsonl(path: Path):
    """Yield dicts from a JSONL file one at a time (memory-safe)."""
    if not path.exists():
        return
    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                try:
                    yield json.loads(line)
                except json.JSONDecodeError:
                    continue


def build_gene_summaries(pubmed_records: List[dict]) -> Dict[str, dict]:
    """Build gene_summaries from PubMed records (first record per gene as seed)."""
    summaries: Dict[str, dict] = {}
    for rec in pubmed_records:
        gene = rec.get("gene", "")
        if not gene or gene in summaries:
            continue
        summaries[gene] = {
            "gene_id": "",
            "symbol": gene,
            "name": gene,
            "summary": rec.get("title", ""),
            "aliases": [],
        }
    return summaries


def build_string_ppi(data_dir: Path, max_partners_per_gene: int = 80) -> Dict[str, list]:
    """Build string_ppi dict: gene -> [{partner, score}, ...].

    For large STRING corpora, keeps only top-scoring partners per gene to keep
    runtime memory bounded while preserving high-confidence neighborhood signal.
    """
    partner_scores: Dict[str, Dict[str, int]] = defaultdict(dict)
    path = data_dir / "string_ppi.jsonl"
    count = 0
    for rec in stream_jsonl(path):
        gene = rec.get("gene", "")
        partner = rec.get("partner", "")
        try:
            score = int(rec.get("score", 0))
        except (TypeError, ValueError):
            score = 0
        if not gene or not partner:
            continue
        cur = partner_scores[gene].get(partner)
        if cur is None or score > cur:
            partner_scores[gene][partner] = score
            count += 1
            if max_partners_per_gene > 0 and len(partner_scores[gene]) > max_partners_per_gene * 2:
                top = sorted(
                    partner_scores[gene].items(),
                    key=lambda kv: kv[1],
                    reverse=True,
                )[:max_partners_per_gene]
                partner_scores[gene] = dict(top)

    ppi: Dict[str, List[dict]] = {}
    kept = 0
    for gene, ps in partner_scores.items():
        top = sorted(ps.items(), key=lambda kv: kv[1], reverse=True)
        if max_partners_per_gene > 0:
            top = top[:max_partners_per_gene]
        ppi[gene] = [{"partner": partner, "score": score} for partner, score in top]
        kept += len(ppi[gene])
    print(f"  [ppi] Processed {count:,} updates, kept {kept:,} interactions for {len(ppi):,} genes")
    return ppi


def _normalize_drug_status(status_list: list) -> list:
    """Normalise ChEMBL max_phase strings to human-readable status.

    ChEMBL returns max_phase as a float (e.g. 4.0, 3.0) or string.
    Phase 4 == approved in ChEMBL terminology.
    """
    result = []
    for s in status_list:
        s_str = str(s).strip().lower()
        # Handle float-encoded phases like "phase 4.0", "4.0", "4"
        phase_num = None
        phase_match = re.match(r"^(?:phase\s*)?([\d.]+)$", s_str)
        if phase_match:
            try:
                phase_num = float(phase_match.group(1))
            except ValueError:
                pass
        if phase_num is not None:
            if phase_num >= 4.0:
                result.append("approved")
            elif phase_num >= 1.0:
                result.append(f"phase {int(phase_num)}")
            # phase < 1 (e.g. -1.0) → skip or mark investigational
            else:
                result.append("investigational")
        elif s_str in ("approved", "investigational", "withdrawn", "experimental"):
            result.append(s_str)
        else:
            result.append(s_str)
    return result if result else ["investigational"]


def _drug_priority(drug: dict) -> int:
    """Return sort priority for a drug record (lower = higher priority)."""
    status = [s.lower() for s in drug.get("status", [])]
    if "approved" in status:
        return 0
    for s in status:
        m = re.match(r"phase (\d+)", s)
        if m:
            return 5 - int(m.group(1))  # phase 4 → 1, phase 3 → 2, etc.
    return 10  # investigational / unknown


def build_drugbank(data_dir: Path) -> Dict[str, list]:
    """Build drugbank dict: gene -> [{drug_name, drugbank_id, status, mechanism}, ...]

    Normalises ChEMBL phase values and sorts by clinical priority
    (approved > phase 4 > phase 3 > … > investigational).
    """
    import re as _re
    drugs: Dict[str, List[dict]] = defaultdict(list)
    path = data_dir / "drugbank_targets.jsonl"
    count = 0
    for rec in stream_jsonl(path):
        gene = rec.get("gene", "")
        if gene:
            normalised_status = _normalize_drug_status(rec.get("status", []))
            drugs[gene].append({
                "drug_name": rec.get("drug_name", ""),
                "drugbank_id": rec.get("drugbank_id", ""),
                "status": normalised_status,
                "mechanism": rec.get("mechanism", ""),
            })
            count += 1
    # Sort each gene's drugs by clinical priority
    for gene in drugs:
        drugs[gene].sort(key=_drug_priority)
    print(f"  [drugbank] Loaded {count:,} drug-target pairs for {len(drugs):,} genes")
    approved_count = sum(1 for gene_drugs in drugs.values()
                         for d in gene_drugs if "approved" in d.get("status", []))
    print(f"  [drugbank] Approved entries: {approved_count:,}")
    return dict(drugs)


def build_go_terms(data_dir: Path) -> Dict[str, dict]:
    """Build go_terms dict: GO_ID -> {id, name, namespace, ...}"""
    terms: Dict[str, dict] = {}
    path = data_dir / "go_terms.jsonl"
    for rec in stream_jsonl(path):
        go_id = rec.get("id", "")
        if go_id:
            terms[go_id] = rec
    print(f"  [go] Loaded {len(terms):,} GO terms")
    return terms


def build_gene_annotations(data_dir: Path) -> Dict[str, list]:
    """Build gene_annotations dict: gene -> [GO_ID, ...]"""
    annots: Dict[str, list] = {}
    path = data_dir / "gene_annotations.jsonl"
    for rec in stream_jsonl(path):
        gene = rec.get("gene", "")
        go_ids = rec.get("go_ids", [])
        if gene and go_ids:
            annots[gene] = go_ids
    print(f"  [go-annot] Loaded annotations for {len(annots):,} genes")
    return annots


def build_gene_synonyms(gene_summaries: Dict[str, dict]) -> Dict[str, str]:
    """Build gene_synonyms from gene summaries."""
    synonyms: Dict[str, str] = {}
    for symbol, info in gene_summaries.items():
        synonyms[symbol] = symbol
        for alias in info.get("aliases", []):
            if alias:
                synonyms[alias] = symbol
    return synonyms


def embed_and_build_faiss(pubmed_records: List[dict], output_dir: Path,
                           batch_size: int = 512, embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"):
    """Embed PubMed abstracts in batches and build a FAISS index.

    Returns the FAISS index path (or None if FAISS is unavailable).
    """
    try:
        import numpy as np
    except ImportError:
        print("  [warn] numpy not available -- skipping FAISS index build")
        return None

    try:
        from sentence_transformers import SentenceTransformer
    except ImportError:
        print("  [warn] sentence-transformers not available -- skipping FAISS index build")
        return None

    try:
        import faiss
    except ImportError:
        print("  [warn] faiss-cpu not available -- skipping FAISS index build")
        return None

    print(f"  [embed] Loading model: {embedding_model}")
    model = SentenceTransformer(embedding_model)

    texts = []
    for rec in pubmed_records:
        text = " ".join([
            str(rec.get("title", "")),
            str(rec.get("abstract", "")),
            str(rec.get("gene", "")),
        ]).strip()
        texts.append(text)

    total = len(texts)
    print(f"  [embed] Encoding {total:,} texts in batches of {batch_size}")

    all_embeddings = []
    for start in range(0, total, batch_size):
        if _shutdown:
            break
        end = min(start + batch_size, total)
        batch_texts = texts[start:end]
        embeddings = model.encode(batch_texts, normalize_embeddings=True,
                                   show_progress_bar=False, batch_size=batch_size)
        all_embeddings.append(np.array(embeddings, dtype=np.float32))
        print(f"    Embedded {end:,}/{total:,}")

    if not all_embeddings:
        return None

    matrix = np.vstack(all_embeddings)
    del all_embeddings
    gc.collect()

    dim = matrix.shape[1]
    n_records = matrix.shape[0]
    print(f"  [faiss] Building index: {n_records:,} vectors, dim={dim}")

    if n_records > 10_000:
        # Use IVF + PQ for large datasets
        nlist = min(int(n_records ** 0.5), 256)
        m = min(dim, 32)  # PQ sub-quantizers
        # Ensure m divides dim
        while dim % m != 0 and m > 1:
            m -= 1
        quantizer = faiss.IndexFlatIP(dim)
        index = faiss.IndexIVFPQ(quantizer, dim, nlist, m, 8)
        print(f"  [faiss] Training IndexIVFPQ (nlist={nlist}, m={m}) ...")
        # Need to train on a sample
        train_size = min(n_records, nlist * 40)
        index.train(matrix[:train_size])
        index.add(matrix)
        index_type = "IndexIVFPQ"
    else:
        index = faiss.IndexFlatIP(dim)
        index.add(matrix)
        index_type = "IndexFlatIP"

    faiss_path = output_dir / "faiss_index.bin"
    faiss.write_index(index, str(faiss_path))
    print(f"  [faiss] Saved {index_type} with {index.ntotal:,} vectors to {faiss_path}")

    del matrix
    gc.collect()
    return faiss_path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build FAISS index and unified demo_bundle.json from downloaded data.")
    parser.add_argument("--data-dir", type=str, default="data/",
                        help="Directory containing JSONL data files")
    parser.add_argument("--output-dir", type=str, default="checkpoints/",
                        help="Directory for output files")
    parser.add_argument("--batch-size", type=int, default=512,
                        help="Embedding batch size")
    parser.add_argument("--embedding-model", type=str,
                        default="sentence-transformers/all-MiniLM-L6-v2",
                        help="Sentence-transformers model for embedding")
    parser.add_argument("--skip-faiss", action="store_true",
                        help="Skip FAISS index building")
    parser.add_argument(
        "--max-partners-per-gene",
        type=int,
        default=80,
        help="Keep top N STRING partners per gene in bundle (0 = keep all)",
    )
    parser.add_argument(
        "--max-pubmed-records",
        type=int,
        default=0,
        help="Optional cap on PubMed records loaded into bundle (0 = all).",
    )
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    start_time = time.time()

    # --- Step 1: Load PubMed records ---
    print("[1/6] Loading PubMed records ...")
    pubmed_path = data_dir / "pubmed_abstracts.jsonl"
    pubmed_records = load_jsonl(pubmed_path)
    if args.max_pubmed_records and args.max_pubmed_records > 0:
        pubmed_records = pubmed_records[: args.max_pubmed_records]
    print(f"  Loaded {len(pubmed_records):,} PubMed records")

    # --- Step 2: Build gene summaries ---
    print("[2/6] Building gene summaries ...")
    gene_summaries = build_gene_summaries(pubmed_records)
    print(f"  {len(gene_summaries):,} gene summaries")

    # --- Step 3: Build PPI, drugbank, GO ---
    print("[3/6] Loading STRING PPI ...")
    string_ppi = build_string_ppi(data_dir, max_partners_per_gene=args.max_partners_per_gene)

    print("[4/6] Loading drug-target data ...")
    drugbank = build_drugbank(data_dir)

    print("[5/6] Loading GO terms and annotations ...")
    go_terms = build_go_terms(data_dir)
    gene_annotations = build_gene_annotations(data_dir)

    # --- Step 4: Build synonyms ---
    gene_synonyms = build_gene_synonyms(gene_summaries)
    print(f"  {len(gene_synonyms):,} gene synonyms")

    # --- Step 5: Build FAISS index ---
    faiss_path = None
    if not args.skip_faiss and pubmed_records:
        print("[6/6] Building FAISS index ...")
        faiss_path = embed_and_build_faiss(
            pubmed_records, output_dir,
            batch_size=args.batch_size,
            embedding_model=args.embedding_model,
        )
    else:
        print("[6/6] Skipping FAISS index build")

    # --- Step 6: Build unified demo_bundle.json ---
    print("\n[bundle] Assembling demo_bundle.json ...")
    bundle = {
        "gene_summaries": gene_summaries,
        "pubmed_records": pubmed_records,
        "string_ppi": string_ppi,
        "drugbank": drugbank,
        "pathways": {},           # Pathways are not ingested by this pipeline
        "pathway_membership": {}, # (can be extended with KEGG ingest later)
        "go_terms": go_terms,
        "gene_annotations": gene_annotations,
        "gene_synonyms": gene_synonyms,
    }

    bundle_path = output_dir / "demo_bundle.json"
    with open(bundle_path, "w", encoding="utf-8") as fh:
        json.dump(bundle, fh, ensure_ascii=False, indent=None)

    elapsed = time.time() - start_time
    peak_mem = _peak_memory_mb()

    print(f"\n=== Build complete in {elapsed:.1f}s ===")
    print(f"  Bundle:       {bundle_path.resolve()}")
    if faiss_path:
        print(f"  FAISS index:  {faiss_path.resolve()}")
    print(f"  PubMed:       {len(pubmed_records):,} records")
    print(f"  Genes:        {len(gene_summaries):,}")
    print(f"  PPI pairs:    {sum(len(v) for v in string_ppi.values()):,}")
    print(f"  Drug-targets: {sum(len(v) for v in drugbank.values()):,}")
    print(f"  GO terms:     {len(go_terms):,}")
    print(f"  Gene annots:  {len(gene_annotations):,}")
    if peak_mem > 0:
        print(f"  Peak memory:  {peak_mem:.0f} MB")
    else:
        print(f"  Peak memory:  (not available on this platform)")


if __name__ == "__main__":
    main()
