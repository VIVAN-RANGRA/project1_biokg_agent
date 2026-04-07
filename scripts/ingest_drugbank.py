"""Build drug-target lookup table.

Usage:
    # If you have DrugBank XML:
    python scripts/ingest_drugbank.py --drugbank-xml data/drugbank.xml --output data/drugbank_targets.jsonl

    # Without DrugBank XML (uses OpenTargets API):
    python scripts/ingest_drugbank.py --use-opentargets --genes 50 --output data/drugbank_targets.jsonl
"""

from __future__ import annotations

import argparse
import json
import os
import signal
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional
from xml.etree import ElementTree

import requests

# ---------------------------------------------------------------------------
# Top 50 biology / cancer genes (same as ingest_pubmed.py)
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

# DrugBank XML namespace
DB_NS = "{http://www.drugbank.ca}"

CHEMBL_API_BASE = "https://www.ebi.ac.uk/chembl/api/data"

_shutdown = False


def _handle_signal(signum, frame):
    global _shutdown
    _shutdown = True
    print("\n[!] Ctrl+C detected -- saving partial progress ...")


signal.signal(signal.SIGINT, _handle_signal)


def _request_with_retry(url: str, json_body: dict | None = None, timeout: int = 30,
                         retries: int = 3, delay: float = 1.0) -> requests.Response:
    """POST (or GET) with exponential back-off retries."""
    last_exc = None
    for attempt in range(retries):
        try:
            if json_body is not None:
                resp = requests.post(url, json=json_body, timeout=timeout)
            else:
                resp = requests.get(url, timeout=timeout)
            resp.raise_for_status()
            return resp
        except (requests.RequestException, requests.HTTPError) as exc:
            last_exc = exc
            wait = delay * (2 ** attempt)
            print(f"  [retry {attempt + 1}/{retries}] {exc} -- waiting {wait:.1f}s")
            time.sleep(wait)
    raise RuntimeError(f"Failed after {retries} retries: {last_exc}")


# ---------------------------------------------------------------------------
# Path A: DrugBank XML (iterparse, memory-safe)
# ---------------------------------------------------------------------------
def parse_drugbank_xml(xml_path: str, output_path: Path) -> int:
    """Parse DrugBank full database XML using iterparse (memory-safe).

    Each <drug> element is processed and cleared to keep memory flat.
    """
    print(f"[drugbank-xml] Parsing {xml_path} ...")
    written = 0

    with open(output_path, "w", encoding="utf-8") as out_fh:
        context = ElementTree.iterparse(xml_path, events=("end",))
        for event, elem in context:
            if _shutdown:
                break
            if elem.tag != f"{DB_NS}drug" and elem.tag != "drug":
                continue

            # Extract DrugBank ID
            db_id_el = elem.find(f"{DB_NS}drugbank-id[@primary='true']")
            if db_id_el is None:
                db_id_el = elem.find(f"{DB_NS}drugbank-id")
            if db_id_el is None:
                db_id_el = elem.find("drugbank-id")
            drugbank_id = db_id_el.text.strip() if db_id_el is not None and db_id_el.text else ""

            # Drug name
            name_el = elem.find(f"{DB_NS}name") or elem.find("name")
            drug_name = name_el.text.strip() if name_el is not None and name_el.text else ""

            # Status / groups
            groups_el = elem.find(f"{DB_NS}groups") or elem.find("groups")
            status: List[str] = []
            if groups_el is not None:
                for g in groups_el:
                    if g.text:
                        status.append(g.text.strip())

            # Mechanism of action
            moa_el = elem.find(f"{DB_NS}mechanism-of-action") or elem.find("mechanism-of-action")
            mechanism = moa_el.text.strip() if moa_el is not None and moa_el.text else ""

            # Targets
            targets_el = elem.find(f"{DB_NS}targets") or elem.find("targets")
            if targets_el is not None:
                for target in targets_el:
                    gene_name_el = target.find(f".//{DB_NS}gene-name") or target.find(".//gene-name")
                    if gene_name_el is not None and gene_name_el.text:
                        gene = gene_name_el.text.strip()
                        action_els = target.findall(f".//{DB_NS}action") or target.findall(".//action")
                        actions = [a.text.strip() for a in action_els if a.text]
                        record = {
                            "gene": gene,
                            "drug_name": drug_name.lower(),
                            "drugbank_id": drugbank_id,
                            "status": status,
                            "mechanism": mechanism or ("; ".join(actions) if actions else ""),
                        }
                        out_fh.write(json.dumps(record, ensure_ascii=False) + "\n")
                        written += 1

            # Free memory -- critical for large XML
            elem.clear()

            if written % 1000 == 0 and written > 0:
                print(f"  [drugbank-xml] {written:,} drug-target pairs written")

    return written


# ---------------------------------------------------------------------------
# Path B: ChEMBL REST API (free, no auth needed, stable schema)
# ---------------------------------------------------------------------------

# Pre-resolved ChEMBL target IDs for the top 50 genes (SINGLE PROTEIN, Homo sapiens)
GENE_TO_CHEMBL_TARGET: Dict[str, str] = {
    "TP53": "CHEMBL4096",    "BRCA1": "CHEMBL4302",   "EGFR": "CHEMBL203",
    "KRAS": "CHEMBL2111425", "MYC": "CHEMBL5384",     "PTEN": "CHEMBL4835",
    "AKT1": "CHEMBL4145",   "BRAF": "CHEMBL5145",     "PIK3CA": "CHEMBL4005",
    "RB1": "CHEMBL5689",    "ERBB2": "CHEMBL1824",    "ALK": "CHEMBL4247",
    "VEGFA": "CHEMBL1783",  "MTOR": "CHEMBL2842",     "CDK4": "CHEMBL4182",
    "ATM": "CHEMBL4439",    "CHEK2": "CHEMBL4444",    "PALB2": "CHEMBL5543",
    "RAD51": "CHEMBL4467",  "MDM2": "CHEMBL3868",     "JAK2": "CHEMBL2971",
    "FLT3": "CHEMBL1974",   "ABL1": "CHEMBL1862",     "BCL2": "CHEMBL4860",
    "NOTCH1": "CHEMBL5543", "CTNNB1": "CHEMBL4909",   "APC": "CHEMBL5553",
    "VHL": "CHEMBL4934",    "IDH1": "CHEMBL3776",     "IDH2": "CHEMBL3612",
    "NRAS": "CHEMBL4765",   "MAP2K1": "CHEMBL3236",   "RAF1": "CHEMBL4807",
    "SRC": "CHEMBL267",     "FGFR1": "CHEMBL3650",    "FGFR2": "CHEMBL2186",
    "FGFR3": "CHEMBL2883",  "MET": "CHEMBL3717",      "RET": "CHEMBL5543",
    "ROS1": "CHEMBL5109",   "KIT": "CHEMBL1955",      "PDGFRA": "CHEMBL1913",
    "SMAD4": "CHEMBL5509",  "STK11": "CHEMBL4481",    "NF1": "CHEMBL5564",
    "NF2": "CHEMBL5498",    "TSC1": "CHEMBL5567",     "TSC2": "CHEMBL5570",
    "WT1": "CHEMBL5587",    "PTCH1": "CHEMBL5573",
}


def _get_chembl_target_id(gene_symbol: str) -> str | None:
    """Resolve gene symbol to ChEMBL target ID via ChEMBL REST API.
    Falls back to pre-resolved table first (faster, no network).
    """
    # Try pre-resolved table first
    if gene_symbol in GENE_TO_CHEMBL_TARGET:
        return GENE_TO_CHEMBL_TARGET[gene_symbol]
    # Dynamic lookup via ChEMBL gene-symbol search
    try:
        url = f"{CHEMBL_API_BASE}/target.json"
        resp = requests.get(
            url,
            params={"target_synonym__iexact": gene_symbol, "organism": "Homo sapiens",
                    "target_type": "SINGLE PROTEIN", "format": "json", "limit": 1},
            timeout=15,
        )
        resp.raise_for_status()
        targets = resp.json().get("targets", [])
        if targets:
            return targets[0]["target_chembl_id"]
    except Exception:
        pass
    return None


def query_chembl(gene_symbol: str) -> List[dict]:
    """Query ChEMBL REST API for approved/clinical drugs targeting a gene."""
    target_id = _get_chembl_target_id(gene_symbol)
    if not target_id:
        print(f"  [skip] No ChEMBL target ID for {gene_symbol}")
        return []

    results: List[dict] = []
    seen: set = set()
    offset = 0
    limit = 100

    while True:
        if _shutdown:
            break
        try:
            resp = requests.get(
                f"{CHEMBL_API_BASE}/activity.json",
                params={
                    "target_chembl_id": target_id,
                    "assay_type": "B",          # binding assays
                    "pchembl_value__isnull": False,
                    "format": "json",
                    "limit": limit,
                    "offset": offset,
                },
                timeout=20,
            )
            resp.raise_for_status()
            data = resp.json()
        except Exception as exc:
            print(f"  [error] ChEMBL activity query failed for {gene_symbol}: {exc}")
            break

        activities = data.get("activities", [])
        if not activities:
            break

        for act in activities:
            mol_id = act.get("molecule_chembl_id") or ""
            mol_name = (act.get("molecule_pref_name") or "").lower()
            if not mol_name or mol_name in seen:
                continue
            seen.add(mol_name)
            results.append({
                "gene": gene_symbol,
                "drug_name": mol_name,
                "drugbank_id": mol_id,   # ChEMBL ID here
                "status": ["investigational"],  # enriched below
                "mechanism": "",
            })

        total = data.get("page_meta", {}).get("total_count", 0)
        offset += limit
        if offset >= min(total, 500):  # cap at 500 activities per gene
            break
        time.sleep(0.2)

    # Enrich top results with approval status from ChEMBL molecule endpoint
    enriched: List[dict] = []
    for rec in results[:50]:  # only enrich top 50 to avoid too many requests
        mol_id = rec["drugbank_id"]
        if not mol_id:
            enriched.append(rec)
            continue
        try:
            mol_resp = requests.get(
                f"{CHEMBL_API_BASE}/molecule/{mol_id}.json",
                params={"format": "json"},
                timeout=10,
            )
            mol_resp.raise_for_status()
            mol = mol_resp.json()
            phase = mol.get("max_phase")
            # ChEMBL returns max_phase as int, float, or string; phase 4 == approved
            phase_num = None
            try:
                phase_num = float(phase) if phase is not None else None
            except (TypeError, ValueError):
                pass
            if phase == "APPROVAL" or (phase_num is not None and phase_num >= 4.0):
                rec["status"] = ["approved"]
            elif phase_num is not None and phase_num >= 1.0:
                rec["status"] = [f"phase {int(phase_num)}"]
            rec["mechanism"] = mol.get("mechanism_of_action") or ""
        except Exception:
            pass
        enriched.append(rec)
        time.sleep(0.1)

    return enriched


def ingest_chembl(genes: List[str], output_path: Path) -> int:
    """Query ChEMBL for all genes and write JSONL."""
    written = 0
    with open(output_path, "w", encoding="utf-8") as out_fh:
        for idx, gene in enumerate(genes):
            if _shutdown:
                break
            print(f"  [{idx + 1}/{len(genes)}] Querying ChEMBL for {gene} ...")
            records = query_chembl(gene)
            for rec in records:
                out_fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
                written += 1
            out_fh.flush()
            print(f"    -> {len(records)} drug records")
            time.sleep(0.5)
    return written


def main() -> None:
    parser = argparse.ArgumentParser(description="Build drug-target lookup table.")
    parser.add_argument("--drugbank-xml", type=str, default=None,
                        help="Path to DrugBank full database XML")
    parser.add_argument("--use-opentargets", action="store_true",
                        help="Use ChEMBL API for drug-target data (--use-opentargets flag kept for compatibility)")
    parser.add_argument("--genes", type=int, default=50,
                        help="Number of top genes to query (OpenTargets mode)")
    parser.add_argument("--output", type=str, default="data/drugbank_targets.jsonl",
                        help="Output JSONL path")
    args = parser.parse_args()

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if args.drugbank_xml:
        if not Path(args.drugbank_xml).exists():
            print(f"[error] DrugBank XML not found: {args.drugbank_xml}")
            sys.exit(1)
        written = parse_drugbank_xml(args.drugbank_xml, output_path)
    elif args.use_opentargets:
        genes = TOP_GENES[: args.genes]
        print(f"[chembl] Querying drugs for {len(genes)} genes via ChEMBL REST API ...")
        written = ingest_chembl(genes, output_path)
    else:
        print("[error] Specify either --drugbank-xml or --use-opentargets")
        sys.exit(1)

    print(f"\n=== Done! {written:,} drug-target records written ===")
    print(f"Output: {output_path.resolve()}")


if __name__ == "__main__":
    main()
