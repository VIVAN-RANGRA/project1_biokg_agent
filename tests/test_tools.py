"""Tests for biokg_agent.tools (offline only -- mock all network calls)."""
import pytest
from unittest.mock import patch, MagicMock


class TestUniprotProteinLookup:
    def test_successful_lookup(self):
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.raise_for_status = MagicMock()
        mock_response.json.return_value = {
            "results": [
                {
                    "primaryAccession": "P04637",
                    "proteinDescription": {
                        "recommendedName": {
                            "fullName": {"value": "Cellular tumor antigen p53"}
                        }
                    },
                    "comments": [
                        {
                            "commentType": "FUNCTION",
                            "texts": [{"value": "Acts as a tumor suppressor"}],
                        }
                    ],
                    "features": [],
                    "uniProtKBCrossReferences": [],
                }
            ]
        }
        with patch("biokg_agent.tools.uniprot.requests.get", return_value=mock_response):
            with patch("biokg_agent.tools.uniprot.time.sleep"):
                from biokg_agent.tools.uniprot import uniprot_protein_lookup
                result = uniprot_protein_lookup("TP53")
        assert result["accession"] == "P04637"
        assert result["name"] == "Cellular tumor antigen p53"
        assert "tumor suppressor" in result["function"]

    def test_api_failure_returns_fallback(self):
        import requests as req_module
        mock_response = MagicMock()
        mock_response.raise_for_status.side_effect = req_module.exceptions.RequestException("fail")
        with patch("biokg_agent.tools.uniprot.requests.get", return_value=mock_response):
            with patch("biokg_agent.tools.uniprot.time.sleep"):
                from biokg_agent.tools.uniprot import uniprot_protein_lookup
                result = uniprot_protein_lookup("TP53")
        assert "error" in result
        assert result["accession"] == ""

    def test_empty_protein_name(self):
        from biokg_agent.tools.uniprot import uniprot_protein_lookup
        result = uniprot_protein_lookup("")
        assert "error" in result

    def test_no_results(self):
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.raise_for_status = MagicMock()
        mock_response.json.return_value = {"results": []}
        with patch("biokg_agent.tools.uniprot.requests.get", return_value=mock_response):
            with patch("biokg_agent.tools.uniprot.time.sleep"):
                from biokg_agent.tools.uniprot import uniprot_protein_lookup
                result = uniprot_protein_lookup("NONEXISTENT_PROTEIN")
        assert "error" in result

    def test_timeout(self):
        import requests as req_module
        with patch("biokg_agent.tools.uniprot.requests.get", side_effect=req_module.exceptions.Timeout("timeout")):
            with patch("biokg_agent.tools.uniprot.time.sleep"):
                from biokg_agent.tools.uniprot import uniprot_protein_lookup
                result = uniprot_protein_lookup("TP53")
        assert "error" in result
        assert "timed out" in result["error"].lower()


class TestKeggPathwayLookup:
    def test_successful_lookup(self):
        def mock_get(url, **kwargs):
            resp = MagicMock()
            resp.status_code = 200
            resp.raise_for_status = MagicMock()
            if "find/genes" in url:
                resp.text = "hsa:7157\tTP53, BCC7, LFS1; tumor protein p53"
            elif "link/pathway" in url:
                resp.text = "hsa:7157\tpath:hsa04115"
            elif "get/" in url:
                resp.text = "NAME        p53 signaling pathway - Homo sapiens (human)\nGENE        7157  TP53; tumor protein p53\n"
            return resp

        with patch("biokg_agent.tools.kegg.requests.get", side_effect=mock_get):
            with patch("biokg_agent.tools.kegg.time.sleep"):
                from biokg_agent.tools.kegg import kegg_pathway_lookup
                result = kegg_pathway_lookup("TP53")
        assert isinstance(result, list)
        assert len(result) > 0
        assert "pathway_id" in result[0]

    def test_empty_gene(self):
        from biokg_agent.tools.kegg import kegg_pathway_lookup
        result = kegg_pathway_lookup("")
        assert result == []

    def test_gene_not_found(self):
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.raise_for_status = MagicMock()
        mock_resp.text = ""
        with patch("biokg_agent.tools.kegg.requests.get", return_value=mock_resp):
            with patch("biokg_agent.tools.kegg.time.sleep"):
                from biokg_agent.tools.kegg import kegg_pathway_lookup
                result = kegg_pathway_lookup("NOTAREALGENE")
        assert result == []


class TestKeggGeneLookup:
    def test_successful_lookup(self):
        def mock_get(url, **kwargs):
            resp = MagicMock()
            resp.status_code = 200
            resp.raise_for_status = MagicMock()
            if "find/genes" in url:
                resp.text = "hsa:7157\tTP53, BCC7; tumor protein p53"
            elif "get/" in url:
                resp.text = (
                    "NAME        TP53\n"
                    "DEFINITION  tumor protein p53\n"
                    "ORTHOLOGY   K04451\n"
                )
            return resp

        with patch("biokg_agent.tools.kegg.requests.get", side_effect=mock_get):
            with patch("biokg_agent.tools.kegg.time.sleep"):
                from biokg_agent.tools.kegg import kegg_gene_lookup
                result = kegg_gene_lookup("TP53")
        assert isinstance(result, dict)
        assert result["kegg_id"] == "hsa:7157"

    def test_empty_gene(self):
        from biokg_agent.tools.kegg import kegg_gene_lookup
        result = kegg_gene_lookup("")
        assert "error" in result


class TestDrugbankTargetLookup:
    def test_opentargets_fallback(self):
        def mock_post(url, **kwargs):
            resp = MagicMock()
            resp.status_code = 200
            resp.raise_for_status = MagicMock()
            body = kwargs.get("json", {})
            query_str = body.get("query", "")
            if "SearchGene" in query_str:
                resp.json.return_value = {
                    "data": {
                        "search": {
                            "hits": [
                                {"id": "ENSG00000141510", "entity": "target", "name": "TP53", "description": ""}
                            ]
                        }
                    }
                }
            elif "DrugsByTarget" in query_str:
                resp.json.return_value = {
                    "data": {
                        "target": {
                            "id": "ENSG00000141510",
                            "approvedSymbol": "TP53",
                            "knownDrugs": {
                                "uniqueDrugs": 1,
                                "rows": [
                                    {
                                        "drug": {
                                            "id": "CHEMBL1234",
                                            "name": "testdrug",
                                            "drugType": "small_molecule",
                                            "maximumClinicalTrialPhase": 4,
                                            "mechanismsOfAction": {
                                                "rows": [{"mechanismOfAction": "p53 activator", "actionType": "activator"}]
                                            },
                                        },
                                        "phase": 4,
                                        "status": "approved",
                                        "urls": [],
                                    }
                                ],
                            },
                        }
                    }
                }
            return resp

        with patch("biokg_agent.tools.drugbank.requests.post", side_effect=mock_post):
            with patch("biokg_agent.tools.drugbank.time.sleep"):
                from biokg_agent.tools.drugbank import drugbank_target_lookup
                result = drugbank_target_lookup("TP53")
        assert isinstance(result, list)
        assert len(result) > 0
        assert result[0]["drug_name"] == "testdrug"

    def test_empty_protein(self):
        from biokg_agent.tools.drugbank import drugbank_target_lookup
        result = drugbank_target_lookup("")
        assert result == []

    def test_api_failure_returns_empty(self):
        import requests as req_module
        with patch("biokg_agent.tools.drugbank.requests.post", side_effect=req_module.exceptions.RequestException("fail")):
            with patch("biokg_agent.tools.drugbank.time.sleep"):
                from biokg_agent.tools.drugbank import drugbank_target_lookup
                result = drugbank_target_lookup("TP53")
        assert result == []


class TestParseDrugbankXml:
    def test_importable(self):
        from biokg_agent.tools.drugbank import parse_drugbank_xml
        assert callable(parse_drugbank_xml)

    def test_file_not_found(self):
        from biokg_agent.tools.drugbank import parse_drugbank_xml
        with pytest.raises(FileNotFoundError):
            parse_drugbank_xml("/nonexistent/path/to/drugbank.xml")
