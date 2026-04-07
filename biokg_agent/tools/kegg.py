"""
KEGG REST API Tool
==================

Provides pathway and gene lookup functionality against the KEGG REST API.
KEGG is freely available for academic use; no API key is required.

API docs: https://www.kegg.jp/kegg/rest/keggapi.html
"""

import os
import re
import time
import logging
from typing import Any, Optional

import requests

__all__ = ["kegg_pathway_lookup", "kegg_gene_lookup"]

logger = logging.getLogger(__name__)

KEGG_BASE_URL = "https://rest.kegg.jp"
REQUEST_TIMEOUT = 15  # seconds
_last_request_time: float = 0.0
_MIN_INTERVAL: float = 0.34  # KEGG asks for max ~3 requests/sec


def _rate_limit() -> None:
    """Enforce a minimum interval between requests to respect KEGG limits."""
    global _last_request_time
    elapsed = time.time() - _last_request_time
    if elapsed < _MIN_INTERVAL:
        time.sleep(_MIN_INTERVAL - elapsed)
    _last_request_time = time.time()


def _kegg_get(path: str) -> Optional[str]:
    """
    Perform a GET against the KEGG REST API and return the response text.

    Returns None on any failure.
    """
    url = f"{KEGG_BASE_URL}/{path.lstrip('/')}"
    try:
        _rate_limit()
        logger.info("KEGG request: %s", url)
        resp = requests.get(url, timeout=REQUEST_TIMEOUT)
        resp.raise_for_status()
        return resp.text
    except requests.exceptions.Timeout:
        logger.warning("KEGG request timed out: %s", url)
        return None
    except requests.exceptions.RequestException as exc:
        logger.warning("KEGG request failed: %s – %s", url, exc)
        return None


# ---------------------------------------------------------------------------
# Gene ID resolution
# ---------------------------------------------------------------------------

def _find_kegg_gene_id(gene_symbol: str) -> Optional[str]:
    """
    Resolve a human gene symbol to a KEGG gene ID (e.g. 'hsa:7157').

    Uses the KEGG /find endpoint to search human genes.
    """
    text = _kegg_get(f"find/genes/{gene_symbol}+homo+sapiens")
    if not text:
        return None

    # Lines look like: "hsa:7157\tTP53, BCC7, LFS1, ...; tumor protein p53"
    for line in text.strip().splitlines():
        parts = line.split("\t")
        if len(parts) < 2:
            continue
        kegg_id = parts[0].strip()
        description = parts[1]
        # Check the gene symbol appears in the alias list (before the semicolon)
        aliases_part = description.split(";")[0] if ";" in description else description
        aliases = [a.strip().upper() for a in aliases_part.split(",")]
        if gene_symbol.upper() in aliases:
            return kegg_id

    # Fallback: return the first hsa entry if exact match not found
    for line in text.strip().splitlines():
        kegg_id = line.split("\t")[0].strip()
        if kegg_id.startswith("hsa:"):
            return kegg_id

    return None


# ---------------------------------------------------------------------------
# Pathway lookup
# ---------------------------------------------------------------------------

def _get_pathways_for_gene(kegg_gene_id: str) -> list[str]:
    """Return a list of KEGG pathway IDs linked to a gene."""
    text = _kegg_get(f"link/pathway/{kegg_gene_id}")
    if not text:
        return []

    pathway_ids: list[str] = []
    for line in text.strip().splitlines():
        parts = line.split("\t")
        if len(parts) >= 2:
            pw_id = parts[1].strip()
            pathway_ids.append(pw_id)
    return pathway_ids


def _parse_pathway_entry(text: str) -> dict[str, Any]:
    """
    Parse KEGG flat-file output for a pathway entry.

    Extracts pathway name and gene list.
    """
    name = ""
    genes: list[str] = []
    in_gene_section = False

    for line in text.splitlines():
        if line.startswith("NAME"):
            name = line[12:].strip()
            # Remove organism suffix like " - Homo sapiens (human)"
            name = re.sub(r"\s*-\s*Homo sapiens.*$", "", name)
        elif line.startswith("GENE"):
            in_gene_section = True
            gene_part = line[12:].strip()
            _parse_gene_line(gene_part, genes)
        elif in_gene_section:
            if line.startswith("            ") or line.startswith("\t"):
                gene_part = line.strip()
                _parse_gene_line(gene_part, genes)
            elif line[0] != " ":
                in_gene_section = False

    return {"pathway_name": name, "genes_in_pathway": genes}


def _parse_gene_line(line: str, genes: list[str]) -> None:
    """Parse a single gene line from KEGG flat-file and append gene symbol."""
    # Format: "7157  TP53; tumor protein p53"
    match = re.match(r"\d+\s+(\S+)", line)
    if match:
        gene_symbol = match.group(1).rstrip(";")
        genes.append(gene_symbol)


def kegg_pathway_lookup(gene: str) -> list[dict[str, Any]]:
    """
    Find KEGG pathways associated with a human gene.

    Steps:
      1. Resolve gene symbol to KEGG gene ID via ``/find/genes/``
      2. Retrieve linked pathways via ``/link/pathway/``
      3. Fetch details for each pathway via ``/get/``

    Parameters
    ----------
    gene : str
        Human gene symbol (e.g. "TP53", "BRCA1", "EGFR").

    Returns
    -------
    list[dict]
        Each dict contains: pathway_id, pathway_name, genes_in_pathway.
        Returns an empty list if no pathways are found or the gene
        cannot be resolved.
    """
    if not gene or not gene.strip():
        return []

    gene = gene.strip().upper()

    # Step 1: Resolve gene symbol to KEGG gene ID
    kegg_gene_id = _find_kegg_gene_id(gene)
    if not kegg_gene_id:
        logger.warning("Could not resolve KEGG gene ID for: %s", gene)
        return []

    logger.info("Resolved %s -> %s", gene, kegg_gene_id)

    # Step 2: Get linked pathways
    pathway_ids = _get_pathways_for_gene(kegg_gene_id)
    if not pathway_ids:
        logger.info("No pathways found for %s", kegg_gene_id)
        return []

    # Step 3: Fetch details for each pathway (limit to first 10 to be polite)
    results: list[dict[str, Any]] = []
    for pw_id in pathway_ids[:10]:
        text = _kegg_get(f"get/{pw_id}")
        if not text:
            results.append({
                "pathway_id": pw_id,
                "pathway_name": "",
                "genes_in_pathway": [],
            })
            continue

        parsed = _parse_pathway_entry(text)
        results.append({
            "pathway_id": pw_id,
            "pathway_name": parsed["pathway_name"],
            "genes_in_pathway": parsed["genes_in_pathway"],
        })

    return results


# ---------------------------------------------------------------------------
# Gene detail lookup
# ---------------------------------------------------------------------------

def kegg_gene_lookup(gene: str) -> dict[str, Any]:
    """
    Retrieve detailed information for a human gene from KEGG.

    Parameters
    ----------
    gene : str
        Human gene symbol (e.g. "TP53").

    Returns
    -------
    dict
        Keys: kegg_id, symbol, name, definition, orthology,
        pathways, diseases, motifs, dblinks.
        Returns a dict with an ``error`` key on failure.
    """
    if not gene or not gene.strip():
        return {"error": "Empty gene name provided", "kegg_id": "", "symbol": gene}

    gene = gene.strip().upper()
    kegg_gene_id = _find_kegg_gene_id(gene)
    if not kegg_gene_id:
        return {"error": f"Could not resolve KEGG gene ID for '{gene}'", "kegg_id": "", "symbol": gene}

    text = _kegg_get(f"get/{kegg_gene_id}")
    if not text:
        return {"error": f"Failed to fetch KEGG entry for {kegg_gene_id}", "kegg_id": kegg_gene_id, "symbol": gene}

    return _parse_gene_entry(text, kegg_gene_id, gene)


def _parse_gene_entry(text: str, kegg_id: str, symbol: str) -> dict[str, Any]:
    """Parse the KEGG flat-file for a gene entry."""
    result: dict[str, Any] = {
        "kegg_id": kegg_id,
        "symbol": symbol,
        "name": "",
        "definition": "",
        "orthology": "",
        "pathways": [],
        "diseases": [],
        "dblinks": {},
    }

    current_section = ""

    for line in text.splitlines():
        # Detect section headers (start at column 0, all caps)
        if line and not line[0].isspace():
            header = line.split()[0] if line.split() else ""
            current_section = header

        if line.startswith("NAME"):
            result["name"] = line[12:].strip()
        elif line.startswith("DEFINITION"):
            result["definition"] = line[12:].strip()
        elif line.startswith("ORTHOLOGY"):
            result["orthology"] = line[12:].strip()
        elif current_section == "PATHWAY":
            content = line[12:].strip() if len(line) > 12 else line.strip()
            if content and not content.startswith("PATHWAY"):
                result["pathways"].append(content)
            elif line.startswith("PATHWAY"):
                content = line[12:].strip()
                if content:
                    result["pathways"].append(content)
        elif current_section == "DISEASE":
            content = line[12:].strip() if len(line) > 12 else line.strip()
            if content and not content.startswith("DISEASE"):
                result["diseases"].append(content)
            elif line.startswith("DISEASE"):
                content = line[12:].strip()
                if content:
                    result["diseases"].append(content)
        elif current_section == "DBLINKS":
            content = line[12:].strip() if len(line) > 12 else line.strip()
            if ":" in content:
                db, ids = content.split(":", 1)
                result["dblinks"][db.strip()] = ids.strip()

    return result


if __name__ == "__main__":
    import json

    print("=" * 60)
    print("KEGG Pathway Lookup: TP53")
    print("=" * 60)
    pathways = kegg_pathway_lookup("TP53")
    for pw in pathways:
        print(f"  {pw['pathway_id']}: {pw['pathway_name']} ({len(pw['genes_in_pathway'])} genes)")

    print(f"\n{'='*60}")
    print("KEGG Gene Lookup: BRCA1")
    print("=" * 60)
    gene_info = kegg_gene_lookup("BRCA1")
    print(json.dumps(gene_info, indent=2, default=str))
