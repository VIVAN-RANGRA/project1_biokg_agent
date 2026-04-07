"""Ingest approved drug-target relationships from ChEMBL (no Open Targets).

This script builds records compatible with ``data/drugbank_targets.jsonl``:
    {"gene": "...", "drug_name": "...", "drugbank_id": "CHEMBL....",
     "status": ["approved"], "mechanism": "..."}

Usage:
    python scripts/ingest_chembl_approved.py --output data/drugbank_targets.jsonl
    python scripts/ingest_chembl_approved.py --target-approved-drugs 2600 --max-molecules 5000
"""

from __future__ import annotations

import argparse
import json
import signal
import time
from pathlib import Path
from typing import Any

import requests

CHEMBL_API_BASE = "https://www.ebi.ac.uk/chembl/api/data"

_shutdown = False


def _handle_signal(signum, frame):
    del signum, frame
    global _shutdown
    _shutdown = True
    print("\n[!] Ctrl+C detected -- stopping after current molecule...")


signal.signal(signal.SIGINT, _handle_signal)


def _get_json(url: str, params: dict[str, Any] | None = None, retries: int = 4, timeout: int = 30) -> dict[str, Any]:
    last_exc: Exception | None = None
    for attempt in range(retries):
        try:
            resp = requests.get(url, params=params, timeout=timeout)
            resp.raise_for_status()
            return resp.json()
        except Exception as exc:  # noqa: BLE001
            last_exc = exc
            wait = min(8.0, 1.2 * (2 ** attempt))
            print(f"  [retry {attempt + 1}/{retries}] {exc} -- waiting {wait:.1f}s")
            time.sleep(wait)
    raise RuntimeError(f"Failed to fetch {url}: {last_exc}")


def _load_existing(output_path: Path) -> tuple[set[tuple[str, str]], set[str]]:
    """Return:
    - seen_pairs: (gene, drug_id) already written
    - seen_drug_ids: unique approved drug IDs already represented
    """
    seen_pairs: set[tuple[str, str]] = set()
    seen_drug_ids: set[str] = set()
    if not output_path.exists():
        return seen_pairs, seen_drug_ids
    with open(output_path, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            gene = str(rec.get("gene", "")).strip().upper()
            drug_id = str(rec.get("drugbank_id", "")).strip()
            if not gene or not drug_id:
                continue
            seen_pairs.add((gene, drug_id))
            statuses = [str(s).lower() for s in rec.get("status", [])]
            if "approved" in statuses:
                seen_drug_ids.add(drug_id)
    return seen_pairs, seen_drug_ids


def _pick_gene_symbol(target_payload: dict[str, Any]) -> str | None:
    """Try to resolve a clean gene symbol from ChEMBL target payload."""
    # Best source: component synonyms with gene-symbol style labels
    for comp in target_payload.get("target_components", []) or []:
        for syn in comp.get("target_component_synonyms", []) or []:
            syn_type = str(syn.get("syn_type", "")).upper()
            sym = str(syn.get("component_synonym", "")).strip().upper()
            if not sym:
                continue
            if "GENE" in syn_type or syn_type in {"HGNC", "GENE_SYMBOL", "HGNC_SYMBOL"}:
                return sym
        # Fallback to component accession if available
        accession = str(comp.get("accession", "")).strip().upper()
        if accession:
            return accession

    # Secondary fallback: target pref name if it looks like a symbol
    pref = str(target_payload.get("pref_name", "")).strip().upper()
    if pref and pref.replace("-", "").replace("_", "").isalnum() and len(pref) <= 15:
        return pref
    return None


def _iter_approved_molecules(page_size: int = 1000):
    offset = 0
    while True:
        if _shutdown:
            return
        payload = _get_json(
            f"{CHEMBL_API_BASE}/molecule.json",
            params={
                "max_phase": 4,
                "format": "json",
                "limit": page_size,
                "offset": offset,
            },
            timeout=40,
        )
        molecules = payload.get("molecules", []) or []
        if not molecules:
            return
        for mol in molecules:
            yield mol
        page_meta = payload.get("page_meta", {}) or {}
        total = int(page_meta.get("total_count", 0) or 0)
        offset += page_size
        print(f"[molecules] fetched {min(offset, total):,}/{total:,}")
        if total and offset >= total:
            return


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build approved drug-target table from ChEMBL (no Open Targets)."
    )
    parser.add_argument("--output", type=str, default="data/drugbank_targets.jsonl", help="Output JSONL path")
    parser.add_argument(
        "--target-approved-drugs",
        type=int,
        default=2600,
        help="Stop once at least this many unique approved drugs have >=1 mapped target gene.",
    )
    parser.add_argument(
        "--max-molecules",
        type=int,
        default=0,
        help="Optional hard cap on number of approved molecules to process (0 = no cap).",
    )
    parser.add_argument("--sleep", type=float, default=0.05, help="Delay between molecule requests (seconds).")
    args = parser.parse_args()

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    seen_pairs, seen_drug_ids = _load_existing(output_path)
    print(f"[resume] Existing unique approved drugs in output: {len(seen_drug_ids):,}")

    target_cache: dict[str, str | None] = {}
    processed_molecules = 0
    written = 0
    mapped_drugs_this_run: set[str] = set()

    with open(output_path, "a", encoding="utf-8") as out_fh:
        for mol in _iter_approved_molecules():
            if _shutdown:
                break
            processed_molecules += 1
            if args.max_molecules and processed_molecules > args.max_molecules:
                break

            drug_id = str(mol.get("molecule_chembl_id", "")).strip()
            if not drug_id:
                continue
            drug_name = str(mol.get("pref_name", "")).strip().lower() or drug_id.lower()

            mech_payload = _get_json(
                f"{CHEMBL_API_BASE}/mechanism.json",
                params={"molecule_chembl_id": drug_id, "format": "json", "limit": 1000},
                timeout=30,
            )
            mechanisms = mech_payload.get("mechanisms", []) or []
            if not mechanisms:
                continue

            has_mapping = False
            for mech in mechanisms:
                raw_target_id = mech.get("target_chembl_id", "")
                target_id = str(raw_target_id).strip()
                if not target_id or target_id.lower() == "none":
                    continue

                if target_id not in target_cache:
                    target_payload = _get_json(
                        f"{CHEMBL_API_BASE}/target/{target_id}.json",
                        params={"format": "json"},
                        timeout=30,
                    )
                    target_cache[target_id] = _pick_gene_symbol(target_payload)

                gene = target_cache[target_id]
                if not gene:
                    continue
                gene = gene.upper()

                pair = (gene, drug_id)
                if pair in seen_pairs:
                    has_mapping = True
                    continue

                action = str(mech.get("action_type", "")).strip()
                moa = str(mech.get("mechanism_of_action", "")).strip()
                mechanism = "; ".join(x for x in [action, moa] if x)

                record = {
                    "gene": gene,
                    "drug_name": drug_name,
                    "drugbank_id": drug_id,
                    "status": ["approved"],
                    "mechanism": mechanism,
                }
                out_fh.write(json.dumps(record, ensure_ascii=False) + "\n")
                seen_pairs.add(pair)
                written += 1
                has_mapping = True

            if has_mapping:
                mapped_drugs_this_run.add(drug_id)
                seen_drug_ids.add(drug_id)

            if processed_molecules % 100 == 0:
                out_fh.flush()
                print(
                    f"[progress] molecules={processed_molecules:,} | "
                    f"new_pairs={written:,} | unique_approved_drugs={len(seen_drug_ids):,}"
                )

            if len(seen_drug_ids) >= args.target_approved_drugs:
                print(
                    f"[target] reached {len(seen_drug_ids):,} unique approved drugs "
                    f"(target {args.target_approved_drugs:,})"
                )
                break

            if args.sleep > 0:
                time.sleep(args.sleep)

    print("\n=== ChEMBL approved ingest complete ===")
    print(f"Processed molecules: {processed_molecules:,}")
    print(f"New gene-drug pairs written: {written:,}")
    print(f"Unique approved drugs represented: {len(seen_drug_ids):,}")
    print(f"Output: {output_path.resolve()}")


if __name__ == "__main__":
    main()
