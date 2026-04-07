"""
UniProt REST API Tool
=====================

Provides protein lookup functionality against the UniProt REST API.
Returns structured protein information including function, GO terms,
domains, PTMs, PDB cross-references, and subcellular location.

API docs: https://www.uniprot.org/help/api
"""

import os
import time
import logging
from typing import Any, Optional

import requests

__all__ = ["uniprot_protein_lookup"]

logger = logging.getLogger(__name__)

UNIPROT_SEARCH_URL = "https://rest.uniprot.org/uniprotkb/search"
REQUEST_TIMEOUT = 15  # seconds
_last_request_time: float = 0.0
_MIN_INTERVAL: float = 0.35  # ~3 req/s to stay within rate limits


def _rate_limit() -> None:
    """Enforce a minimum interval between requests."""
    global _last_request_time
    elapsed = time.time() - _last_request_time
    if elapsed < _MIN_INTERVAL:
        time.sleep(_MIN_INTERVAL - elapsed)
    _last_request_time = time.time()


def _extract_function(entry: dict) -> str:
    """Extract function description from a UniProt JSON entry."""
    comments = entry.get("comments", [])
    for comment in comments:
        if comment.get("commentType") == "FUNCTION":
            texts = comment.get("texts", [])
            if texts:
                return texts[0].get("value", "")
    return ""


def _extract_go_terms(entry: dict) -> list[dict[str, str]]:
    """Extract Gene Ontology terms from cross-references."""
    go_terms: list[dict[str, str]] = []
    xrefs = entry.get("uniProtKBCrossReferences", [])
    for xref in xrefs:
        if xref.get("database") == "GO":
            term_id = xref.get("id", "")
            properties = {
                p.get("key", ""): p.get("value", "")
                for p in xref.get("properties", [])
            }
            go_terms.append({
                "id": term_id,
                "term": properties.get("GoTerm", ""),
                "source": properties.get("GoEvidenceType", ""),
            })
    return go_terms


def _extract_domains(entry: dict) -> list[dict[str, Any]]:
    """Extract domain/region features from the entry."""
    domains: list[dict[str, Any]] = []
    for feature in entry.get("features", []):
        ftype = feature.get("type", "")
        if ftype in ("Domain", "Region", "Repeat", "Motif", "Zinc finger"):
            loc = feature.get("location", {})
            start = loc.get("start", {}).get("value")
            end = loc.get("end", {}).get("value")
            domains.append({
                "type": ftype,
                "description": feature.get("description", ""),
                "start": start,
                "end": end,
            })
    return domains


def _extract_ptms(entry: dict) -> list[dict[str, Any]]:
    """Extract post-translational modification features."""
    ptms: list[dict[str, Any]] = []
    ptm_types = (
        "Modified residue", "Glycosylation", "Disulfide bond",
        "Cross-link", "Lipidation",
    )
    for feature in entry.get("features", []):
        if feature.get("type", "") in ptm_types:
            loc = feature.get("location", {})
            start = loc.get("start", {}).get("value")
            end = loc.get("end", {}).get("value")
            ptms.append({
                "type": feature.get("type", ""),
                "description": feature.get("description", ""),
                "position_start": start,
                "position_end": end,
            })
    return ptms


def _extract_pdb_ids(entry: dict) -> list[str]:
    """Extract PDB cross-reference identifiers."""
    pdb_ids: list[str] = []
    for xref in entry.get("uniProtKBCrossReferences", []):
        if xref.get("database") == "PDB":
            pdb_id = xref.get("id", "")
            if pdb_id:
                pdb_ids.append(pdb_id)
    return pdb_ids


def _extract_subcellular_location(entry: dict) -> list[str]:
    """Extract subcellular location annotations."""
    locations: list[str] = []
    for comment in entry.get("comments", []):
        if comment.get("commentType") == "SUBCELLULAR LOCATION":
            for sc in comment.get("subcellularLocations", []):
                loc = sc.get("location", {})
                value = loc.get("value", "")
                if value:
                    locations.append(value)
    return locations


def _build_fallback(protein_name: str, error_msg: str) -> dict[str, Any]:
    """Return a minimal result dict when the API call fails."""
    return {
        "accession": "",
        "name": protein_name,
        "function": "",
        "go_terms": [],
        "domains": [],
        "ptms": [],
        "pdb_ids": [],
        "subcellular_location": [],
        "error": error_msg,
    }


def uniprot_protein_lookup(protein_name: str) -> dict[str, Any]:
    """
    Look up a human protein in UniProt by name or gene symbol.

    Queries the UniProt REST API for reviewed (Swiss-Prot) human entries
    matching the given protein name, then extracts structured annotations.

    Parameters
    ----------
    protein_name : str
        Protein name or gene symbol (e.g. "TP53", "BRCA1", "insulin").

    Returns
    -------
    dict
        Keys: accession, name, function, go_terms, domains, ptms,
        pdb_ids, subcellular_location.  If the lookup fails, an
        ``error`` key is included and other fields may be empty.
    """
    if not protein_name or not protein_name.strip():
        return _build_fallback(protein_name, "Empty protein name provided")

    query = f'(gene:{protein_name} OR protein_name:{protein_name}) AND organism_id:9606 AND reviewed:true'
    params = {
        "query": query,
        "format": "json",
        "size": "1",
        "fields": (
            "accession,protein_name,gene_names,cc_function,"
            "ft_domain,ft_region,ft_mod_res,ft_carbohyd,ft_disulfid,"
            "ft_lipid,xref_pdb,xref_go,cc_subcellular_location"
        ),
    }

    try:
        _rate_limit()
        logger.info("Querying UniProt for protein: %s", protein_name)
        response = requests.get(
            UNIPROT_SEARCH_URL,
            params=params,
            timeout=REQUEST_TIMEOUT,
            headers={"Accept": "application/json"},
        )
        response.raise_for_status()
    except requests.exceptions.Timeout:
        logger.warning("UniProt request timed out for %s", protein_name)
        return _build_fallback(protein_name, "Request timed out")
    except requests.exceptions.RequestException as exc:
        logger.warning("UniProt request failed for %s: %s", protein_name, exc)
        return _build_fallback(protein_name, f"Request failed: {exc}")

    try:
        data = response.json()
    except ValueError:
        return _build_fallback(protein_name, "Failed to parse JSON response")

    results = data.get("results", [])
    if not results:
        return _build_fallback(protein_name, f"No UniProt entry found for '{protein_name}'")

    entry = results[0]

    # Extract protein recommended name
    protein_desc = entry.get("proteinDescription", {})
    rec_name = protein_desc.get("recommendedName", {})
    full_name = rec_name.get("fullName", {}).get("value", protein_name)

    accession = entry.get("primaryAccession", "")

    return {
        "accession": accession,
        "name": full_name,
        "function": _extract_function(entry),
        "go_terms": _extract_go_terms(entry),
        "domains": _extract_domains(entry),
        "ptms": _extract_ptms(entry),
        "pdb_ids": _extract_pdb_ids(entry),
        "subcellular_location": _extract_subcellular_location(entry),
    }


if __name__ == "__main__":
    import json

    test_proteins = ["TP53", "BRCA1", "INS"]
    for name in test_proteins:
        print(f"\n{'='*60}")
        print(f"Looking up: {name}")
        print("=" * 60)
        result = uniprot_protein_lookup(name)
        print(json.dumps(result, indent=2, default=str))
