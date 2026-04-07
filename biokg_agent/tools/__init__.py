"""
BioKG Agent Tools Module
=========================

API tool wrappers for querying biological databases:
- UniProt (protein information)
- KEGG (pathway and gene information)
- DrugBank / OpenTargets (drug-target associations)
"""

from biokg_agent.tools.uniprot import uniprot_protein_lookup
from biokg_agent.tools.kegg import kegg_pathway_lookup, kegg_gene_lookup
from biokg_agent.tools.drugbank import drugbank_target_lookup, parse_drugbank_xml

__all__ = [
    "uniprot_protein_lookup",
    "kegg_pathway_lookup",
    "kegg_gene_lookup",
    "drugbank_target_lookup",
    "parse_drugbank_xml",
]
