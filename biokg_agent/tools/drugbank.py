"""
DrugBank / OpenTargets Drug-Target Lookup Tool
===============================================

Two-pronged approach for drug-target associations:

1. **Primary**: If the user has a local DrugBank XML/JSON data file
   (path set via the ``DRUGBANK_DATA_PATH`` environment variable),
   parse it directly for drug-target mappings.

2. **Fallback**: Query the free OpenTargets Platform GraphQL API to
   retrieve drug-target associations when local data is unavailable.

DrugBank data download: https://go.drugbank.com/releases/latest
OpenTargets API docs:   https://platform-docs.opentargets.org/
"""

import os
import json
import time
import logging
import xml.etree.ElementTree as ET
from typing import Any, Optional

import requests

__all__ = ["drugbank_target_lookup", "parse_drugbank_xml"]

logger = logging.getLogger(__name__)

OPENTARGETS_GRAPHQL_URL = "https://api.platform.opentargets.org/api/v4/graphql"
REQUEST_TIMEOUT = 15  # seconds
_last_request_time: float = 0.0
_MIN_INTERVAL: float = 0.2  # OpenTargets is generous but be polite


def _rate_limit() -> None:
    """Enforce a minimum interval between requests."""
    global _last_request_time
    elapsed = time.time() - _last_request_time
    if elapsed < _MIN_INTERVAL:
        time.sleep(_MIN_INTERVAL - elapsed)
    _last_request_time = time.time()


# ---------------------------------------------------------------------------
# DrugBank XML Parser
# ---------------------------------------------------------------------------

def parse_drugbank_xml(xml_path: str) -> dict[str, list[dict[str, str]]]:
    """
    Parse a DrugBank full-database XML file and build a target-to-drugs mapping.

    The DrugBank XML uses the namespace
    ``http://www.drugbank.ca``. Each ``<drug>`` element contains
    ``<targets>`` with polypeptide names and actions.

    Parameters
    ----------
    xml_path : str
        Absolute path to the DrugBank XML file (e.g. ``full database.xml``).

    Returns
    -------
    dict[str, list[dict]]
        Mapping from target gene symbol (uppercased) to a list of dicts,
        each with keys: drug_name, drugbank_id, type, mechanism, status.
    """
    if not os.path.isfile(xml_path):
        raise FileNotFoundError(f"DrugBank XML not found: {xml_path}")

    ns = {"db": "http://www.drugbank.ca"}
    target_map: dict[str, list[dict[str, str]]] = {}

    logger.info("Parsing DrugBank XML: %s", xml_path)

    # Use iterparse for memory efficiency on the large XML file
    context = ET.iterparse(xml_path, events=("end",))
    drug_tag = f"{{{ns['db']}}}drug"

    current_drug_name = ""
    current_drug_id = ""
    current_drug_type = ""
    current_groups: list[str] = []

    for event, elem in context:
        if elem.tag == drug_tag and elem.get("type"):
            # Top-level <drug> element
            current_drug_type = elem.get("type", "")

            name_el = elem.find("db:name", ns)
            current_drug_name = name_el.text if name_el is not None and name_el.text else ""

            id_el = elem.find("db:drugbank-id[@primary='true']", ns)
            if id_el is None:
                id_el = elem.find("db:drugbank-id", ns)
            current_drug_id = id_el.text if id_el is not None and id_el.text else ""

            # Approval status
            current_groups = []
            groups_el = elem.find("db:groups", ns)
            if groups_el is not None:
                for g in groups_el.findall("db:group", ns):
                    if g.text:
                        current_groups.append(g.text)

            status = "approved" if "approved" in current_groups else (
                current_groups[0] if current_groups else "unknown"
            )

            # Process targets
            targets_el = elem.find("db:targets", ns)
            if targets_el is not None:
                for target in targets_el.findall("db:target", ns):
                    polypeptide = target.find("db:polypeptide", ns)
                    gene_name = ""
                    if polypeptide is not None:
                        gn_el = polypeptide.find("db:gene-name", ns)
                        gene_name = gn_el.text.upper() if gn_el is not None and gn_el.text else ""

                    actions_el = target.find("db:actions", ns)
                    mechanism_parts: list[str] = []
                    if actions_el is not None:
                        for act in actions_el.findall("db:action", ns):
                            if act.text:
                                mechanism_parts.append(act.text)

                    if gene_name:
                        entry = {
                            "drug_name": current_drug_name,
                            "drugbank_id": current_drug_id,
                            "type": current_drug_type,
                            "mechanism": ", ".join(mechanism_parts) if mechanism_parts else "unknown",
                            "status": status,
                        }
                        target_map.setdefault(gene_name, []).append(entry)

            # Free memory for processed element
            elem.clear()

    logger.info("Parsed %d targets from DrugBank XML", len(target_map))
    return target_map


def _lookup_from_local_drugbank(protein: str) -> Optional[list[dict[str, str]]]:
    """
    Try to look up drug-target associations from a local DrugBank file.

    Checks the ``DRUGBANK_DATA_PATH`` environment variable. Supports
    both XML and pre-processed JSON formats.

    Returns None if local data is not available or the protein is not found.
    """
    data_path = os.environ.get("DRUGBANK_DATA_PATH", "")
    if not data_path or not os.path.isfile(data_path):
        return None

    try:
        if data_path.lower().endswith(".json"):
            with open(data_path, "r", encoding="utf-8") as f:
                db_data = json.load(f)
            # Expect JSON structure: {gene_symbol: [drug_entries]}
            results = db_data.get(protein.upper(), [])
            return results if results else None
        elif data_path.lower().endswith(".xml"):
            target_map = parse_drugbank_xml(data_path)
            results = target_map.get(protein.upper(), [])
            return results if results else None
    except Exception as exc:
        logger.warning("Failed to read local DrugBank data: %s", exc)

    return None


# ---------------------------------------------------------------------------
# OpenTargets GraphQL Fallback
# ---------------------------------------------------------------------------

def _resolve_ensembl_id(gene_symbol: str) -> Optional[str]:
    """
    Resolve a gene symbol to an Ensembl gene ID using the OpenTargets
    search endpoint.
    """
    query = """
    query SearchGene($symbol: String!) {
      search(queryString: $symbol, entityNames: ["target"], page: {index: 0, size: 5}) {
        hits {
          id
          entity
          name
          description
        }
      }
    }
    """
    try:
        _rate_limit()
        resp = requests.post(
            OPENTARGETS_GRAPHQL_URL,
            json={"query": query, "variables": {"symbol": gene_symbol}},
            timeout=REQUEST_TIMEOUT,
            headers={"Content-Type": "application/json"},
        )
        resp.raise_for_status()
        data = resp.json()
    except (requests.exceptions.RequestException, ValueError) as exc:
        logger.warning("OpenTargets search failed for %s: %s", gene_symbol, exc)
        return None

    hits = data.get("data", {}).get("search", {}).get("hits", [])
    for hit in hits:
        if hit.get("entity") == "target":
            ens_id = hit.get("id", "")
            if ens_id.startswith("ENSG"):
                return ens_id

    return None


def _opentargets_drug_lookup(gene_symbol: str) -> list[dict[str, str]]:
    """
    Query the OpenTargets Platform GraphQL API for known drugs
    that interact with a given gene/protein target.
    """
    ensembl_id = _resolve_ensembl_id(gene_symbol)
    if not ensembl_id:
        logger.warning("Could not resolve Ensembl ID for %s", gene_symbol)
        return []

    query = """
    query DrugsByTarget($ensemblId: String!) {
      target(ensemblId: $ensemblId) {
        id
        approvedSymbol
        knownDrugs(size: 25) {
          uniqueDrugs
          rows {
            drug {
              id
              name
              drugType
              maximumClinicalTrialPhase
              mechanismsOfAction {
                rows {
                  mechanismOfAction
                  actionType
                }
              }
            }
            phase
            status
            urls {
              name
              url
            }
          }
        }
      }
    }
    """

    try:
        _rate_limit()
        resp = requests.post(
            OPENTARGETS_GRAPHQL_URL,
            json={"query": query, "variables": {"ensemblId": ensembl_id}},
            timeout=REQUEST_TIMEOUT,
            headers={"Content-Type": "application/json"},
        )
        resp.raise_for_status()
        data = resp.json()
    except (requests.exceptions.RequestException, ValueError) as exc:
        logger.warning("OpenTargets drug query failed for %s: %s", ensembl_id, exc)
        return []

    target_data = data.get("data", {}).get("target", {})
    if not target_data:
        return []

    known_drugs = target_data.get("knownDrugs", {})
    rows = known_drugs.get("rows", [])

    # De-duplicate by drug name
    seen: set[str] = set()
    results: list[dict[str, str]] = []

    for row in rows:
        drug = row.get("drug", {})
        drug_name = drug.get("name", "")
        if not drug_name or drug_name.lower() in seen:
            continue
        seen.add(drug_name.lower())

        drug_id = drug.get("id", "")
        drug_type = drug.get("drugType", "unknown")

        # Extract mechanism of action
        moa_rows = drug.get("mechanismsOfAction", {}).get("rows", [])
        mechanism = ""
        if moa_rows:
            mechanism = moa_rows[0].get("mechanismOfAction", "")

        # Determine clinical status
        max_phase = drug.get("maximumClinicalTrialPhase")
        phase_map = {
            0: "preclinical",
            1: "Phase I",
            2: "Phase II",
            3: "Phase III",
            4: "approved",
        }
        status = phase_map.get(max_phase, f"Phase {max_phase}" if max_phase else "unknown")

        results.append({
            "drug_name": drug_name,
            "drugbank_id": drug_id,  # OpenTargets uses CHEMBL IDs
            "type": drug_type,
            "mechanism": mechanism,
            "status": status,
        })

    return results


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def drugbank_target_lookup(protein: str) -> list[dict[str, str]]:
    """
    Look up drugs that target a given protein/gene.

    Strategy:
      1. If ``DRUGBANK_DATA_PATH`` env var points to a valid local file,
         parse it for drug-target associations.
      2. Otherwise, fall back to the free OpenTargets GraphQL API.

    Parameters
    ----------
    protein : str
        Protein/gene symbol (e.g. "TP53", "EGFR", "BRAF").

    Returns
    -------
    list[dict]
        Each dict contains: drug_name, drugbank_id (or CHEMBL id),
        type, mechanism, status.  Returns an empty list if no
        associations are found.
    """
    if not protein or not protein.strip():
        return []

    protein = protein.strip().upper()

    # Try local DrugBank data first
    local_results = _lookup_from_local_drugbank(protein)
    if local_results is not None:
        logger.info("Found %d drugs for %s from local DrugBank data", len(local_results), protein)
        return local_results

    # Fallback to OpenTargets
    logger.info("Using OpenTargets fallback for drug-target lookup: %s", protein)
    return _opentargets_drug_lookup(protein)


if __name__ == "__main__":
    test_targets = ["EGFR", "BRAF", "TP53"]
    for target in test_targets:
        print(f"\n{'='*60}")
        print(f"Drug-target lookup: {target}")
        print("=" * 60)
        drugs = drugbank_target_lookup(target)
        if not drugs:
            print("  No drugs found.")
        for d in drugs[:5]:
            print(f"  {d['drug_name']} ({d['drugbank_id']}) – {d['mechanism']} [{d['status']}]")
        if len(drugs) > 5:
            print(f"  ... and {len(drugs)-5} more")
