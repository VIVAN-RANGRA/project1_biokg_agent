"""LLM-based answer synthesis and planning for BioKG-Agent.

Supports two backend families:
1. **Groq API** (free tier) -- cloud-based, no GPU needed, fast inference
2. **Local GPU** (Kaggle T4) -- Qwen2.5-7B-Instruct-AWQ via vLLM/AutoAWQ/BNB

The module is importable even when torch/transformers are not installed;
all heavy imports are guarded behind try/except blocks.
"""

from __future__ import annotations

import json
import logging
import os
import re
from typing import Any

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Environment-driven defaults
# ---------------------------------------------------------------------------

DEFAULT_MODEL_ID = os.environ.get(
    "BIOKG_MODEL_ID", "Qwen/Qwen2.5-7B-Instruct-AWQ"
)
DEFAULT_DEVICE = os.environ.get("BIOKG_DEVICE", "auto")
DEFAULT_MAX_TOKENS = int(os.environ.get("BIOKG_MAX_TOKENS", "1024"))

# ---------------------------------------------------------------------------
# System prompt
# ---------------------------------------------------------------------------

BIOKG_SYSTEM_PROMPT = (
    "You are BioKG-Agent, a senior computational biologist assistant. "
    "You answer questions about genes, proteins, pathways, and cancer drugs "
    "by synthesizing evidence from multiple databases.\n\n"
    "STRICT RULES:\n"
    "- Use ONLY the evidence provided in the prompt. Never invent or guess.\n"
    "- Start with a direct 2-3 sentence answer to the question.\n"
    "- If the evidence shows approved drugs, list each drug name and its "
    "mechanism of action in 1 sentence (e.g. 'Olaparib is a PARP inhibitor that...').\n"
    "- Mention 2-3 key protein interactions and explain their biological "
    "significance — do NOT list scores or enumerate every interaction.\n"
    "- Mention the most relevant 1-2 pathways and GO processes in one sentence.\n"
    "- Cite 1-2 literature titles by name to support the answer.\n"
    "- Write in flowing prose paragraphs, NOT bullet lists or score tables.\n"
    "- Keep total response under 300 words.\n"
    "- Do NOT say 'there is limited evidence' if drugs/interactions are "
    "listed — use that evidence.\n"
    "- Do NOT include confidence percentages, STRING scores, DrugBank IDs, "
    "or PMIDs in the answer text — the UI displays these separately."
)

BIOKG_PLANNER_SYSTEM_PROMPT = (
    "You are the query planner for BioKG-Agent. Given a biological question "
    "and a list of known gene symbols, output a JSON object with the "
    "following keys:\n"
    "- query_type: one of 'mechanistic', 'relationship', 'literature', "
    "'entity', 'hybrid'\n"
    "- retrieval_modes: list from ['dense', 'bm25', 'graph']\n"
    "- metadata_filters: dict, typically {\"genes\": [...]}\n"
    "- detected_entities: list of gene symbols found in the query\n"
    "- use_reranker: bool\n"
    "- requires_graph_expansion: bool\n"
    "- requires_api_lookup: bool\n"
    "- max_iterations: int (1-3)\n"
    "- route_confidence: float (0-1)\n"
    "- rationale: short explanation of routing decision\n\n"
    "Return ONLY valid JSON, no markdown fences, no extra text."
)

# ---------------------------------------------------------------------------
# Groq API Backend (free tier, no GPU needed)
# ---------------------------------------------------------------------------

try:
    import requests as _requests
except Exception:
    _requests = None  # type: ignore[assignment]


class GroqBackend:
    """LLM backend using Groq's free API — no GPU required.

    Groq provides extremely fast inference for open models (Llama 3.3 70B,
    Mixtral, Gemma, etc.) with a generous free tier.

    Parameters
    ----------
    api_key : str
        Groq API key (starts with ``gsk_``).
    model : str
        Model identifier on Groq. Default ``llama-3.3-70b-versatile``.
    max_tokens : int
        Maximum generation length.
    """

    API_URL = "https://api.groq.com/openai/v1/chat/completions"

    def __init__(
        self,
        api_key: str,
        model: str = "llama-3.3-70b-versatile",
        max_tokens: int = DEFAULT_MAX_TOKENS,
    ) -> None:
        self.api_key = api_key
        self.model = model
        self.max_tokens = max_tokens
        self._backend = "groq"

    def load(self) -> None:
        """No-op for API backend — validates the key format."""
        if not self.api_key or not self.api_key.startswith("gsk_"):
            logger.warning("Groq API key looks invalid (should start with gsk_)")

    def is_loaded(self) -> bool:
        return bool(self.api_key)

    def generate(
        self,
        prompt: str,
        system_prompt: str = "",
        max_new_tokens: int | None = None,
    ) -> str:
        """Call Groq API for chat completion."""
        if _requests is None:
            raise RuntimeError("requests library is required for GroqBackend")

        messages: list[dict[str, str]] = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        response = _requests.post(
            self.API_URL,
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
            json={
                "model": self.model,
                "messages": messages,
                "max_tokens": max_new_tokens or self.max_tokens,
                "temperature": 0.3,
                "top_p": 0.9,
            },
            timeout=60,
        )
        response.raise_for_status()
        data = response.json()
        return data["choices"][0]["message"]["content"].strip()

    def __repr__(self) -> str:
        return f"GroqBackend(model={self.model!r}, status=ready)"


# ---------------------------------------------------------------------------
# Local GPU Backend (Kaggle T4)
# ---------------------------------------------------------------------------


class LLMBackend:
    """Pluggable LLM backend for BioKG-Agent synthesis and planning.

    Supports three loading strategies (tried in order):
    1. vLLM  -- fastest on Kaggle T4, tensor-parallel ready
    2. transformers + AutoAWQ  -- good AWQ kernel support
    3. transformers + bitsandbytes 4-bit  -- widest compatibility

    The model is lazily loaded on first call to :meth:`generate` or
    explicitly via :meth:`load`.

    Parameters
    ----------
    model_id : str
        HuggingFace model identifier.
    device : str
        Device map string passed to transformers (``"auto"``, ``"cuda"``,
        ``"cpu"``).  Ignored when using vLLM.
    max_new_tokens : int
        Default generation length.
    """

    def __init__(
        self,
        model_id: str = DEFAULT_MODEL_ID,
        device: str = DEFAULT_DEVICE,
        max_new_tokens: int = DEFAULT_MAX_TOKENS,
    ) -> None:
        self.model_id: str = model_id
        self.device: str = device
        self.max_new_tokens: int = max_new_tokens

        # Internals -- populated by load()
        self._model: Any = None
        self._tokenizer: Any = None
        self._backend: str | None = None  # "vllm" | "transformers" | None
        self._sampling_params: Any = None  # vLLM SamplingParams if applicable

    # ------------------------------------------------------------------
    # Loading
    # ------------------------------------------------------------------

    def load(self) -> None:
        """Load model and tokenizer.

        Tries vLLM first, then transformers+AWQ, then transformers+BNB.
        Raises ``RuntimeError`` if none of the backends are available.
        """
        if self._model is not None:
            logger.info("Model already loaded (%s backend).", self._backend)
            return

        if self._try_load_vllm():
            return
        if self._try_load_transformers_awq():
            return
        if self._try_load_transformers_bnb():
            return

        raise RuntimeError(
            "Could not load the LLM. Install at least one of: "
            "vllm, transformers+autoawq, transformers+bitsandbytes. "
            f"Model requested: {self.model_id}"
        )

    def _try_load_vllm(self) -> bool:
        """Attempt to load via vLLM (fastest on T4)."""
        try:
            from vllm import LLM, SamplingParams  # type: ignore[import-untyped]

            logger.info("Loading model via vLLM: %s", self.model_id)
            self._model = LLM(
                model=self.model_id,
                dtype="half",
                quantization="awq",
                max_model_len=4096,
                gpu_memory_utilization=0.85,
            )
            self._sampling_params = SamplingParams(
                max_tokens=self.max_new_tokens,
                temperature=0.3,
                top_p=0.9,
                stop=["<|im_end|>", "<|endoftext|>"],
            )
            self._backend = "vllm"
            logger.info("vLLM backend ready.")
            return True
        except Exception as exc:  # noqa: BLE001
            logger.debug("vLLM not available: %s", exc)
            return False

    def _try_load_transformers_awq(self) -> bool:
        """Attempt to load via transformers with AutoAWQ kernels."""
        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer  # type: ignore[import-untyped]

            logger.info(
                "Loading model via transformers+AWQ: %s", self.model_id
            )
            self._tokenizer = AutoTokenizer.from_pretrained(
                self.model_id, trust_remote_code=True
            )
            self._model = AutoModelForCausalLM.from_pretrained(
                self.model_id,
                device_map=self.device,
                trust_remote_code=True,
            )
            self._backend = "transformers"
            logger.info("transformers+AWQ backend ready.")
            return True
        except Exception as exc:  # noqa: BLE001
            logger.debug("transformers+AWQ not available: %s", exc)
            return False

    def _try_load_transformers_bnb(self) -> bool:
        """Attempt to load via transformers with bitsandbytes 4-bit."""
        try:
            from transformers import (  # type: ignore[import-untyped]
                AutoModelForCausalLM,
                AutoTokenizer,
                BitsAndBytesConfig,
            )

            logger.info(
                "Loading model via transformers+BNB 4-bit: %s", self.model_id
            )
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype="float16",
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
            )
            self._tokenizer = AutoTokenizer.from_pretrained(
                self.model_id, trust_remote_code=True
            )
            self._model = AutoModelForCausalLM.from_pretrained(
                self.model_id,
                quantization_config=bnb_config,
                device_map=self.device,
                trust_remote_code=True,
            )
            self._backend = "transformers"
            logger.info("transformers+BNB 4-bit backend ready.")
            return True
        except Exception as exc:  # noqa: BLE001
            logger.debug("transformers+BNB not available: %s", exc)
            return False

    # ------------------------------------------------------------------
    # Generation
    # ------------------------------------------------------------------

    def generate(
        self,
        prompt: str,
        system_prompt: str = "",
        max_new_tokens: int | None = None,
    ) -> str:
        """Generate text from *prompt*.

        Parameters
        ----------
        prompt : str
            User / instruction text.
        system_prompt : str
            Optional system message prepended in chat format.
        max_new_tokens : int | None
            Override instance default for this call.

        Returns
        -------
        str
            Generated text (assistant turn only).
        """
        if not self.is_loaded():
            self.load()

        max_tokens = max_new_tokens or self.max_new_tokens

        if self._backend == "vllm":
            return self._generate_vllm(prompt, system_prompt, max_tokens)
        return self._generate_transformers(prompt, system_prompt, max_tokens)

    def _build_chat_messages(
        self, prompt: str, system_prompt: str
    ) -> list[dict[str, str]]:
        """Build a Qwen-style chat message list."""
        messages: list[dict[str, str]] = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})
        return messages

    def _generate_vllm(
        self, prompt: str, system_prompt: str, max_tokens: int
    ) -> str:
        from vllm import SamplingParams  # type: ignore[import-untyped]

        messages = self._build_chat_messages(prompt, system_prompt)

        # vLLM accepts chat messages via the tokenizer's chat template
        tokenizer = self._model.get_tokenizer()  # type: ignore[union-attr]
        formatted = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

        params = SamplingParams(
            max_tokens=max_tokens,
            temperature=0.3,
            top_p=0.9,
            stop=["<|im_end|>", "<|endoftext|>"],
        )
        outputs = self._model.generate([formatted], params)  # type: ignore[union-attr]
        return outputs[0].outputs[0].text.strip()

    def _generate_transformers(
        self, prompt: str, system_prompt: str, max_tokens: int
    ) -> str:
        import torch  # type: ignore[import-untyped]

        messages = self._build_chat_messages(prompt, system_prompt)
        text = self._tokenizer.apply_chat_template(  # type: ignore[union-attr]
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = self._tokenizer(text, return_tensors="pt").to(  # type: ignore[union-attr]
            self._model.device  # type: ignore[union-attr]
        )
        with torch.no_grad():
            output_ids = self._model.generate(  # type: ignore[union-attr]
                **inputs,
                max_new_tokens=max_tokens,
                temperature=0.3,
                top_p=0.9,
                do_sample=True,
                pad_token_id=self._tokenizer.eos_token_id,  # type: ignore[union-attr]
            )
        # Decode only the new tokens
        generated_ids = output_ids[0][inputs["input_ids"].shape[1] :]
        return self._tokenizer.decode(  # type: ignore[union-attr]
            generated_ids, skip_special_tokens=True
        ).strip()

    # ------------------------------------------------------------------
    # Status
    # ------------------------------------------------------------------

    def is_loaded(self) -> bool:
        """Return ``True`` if the model has been loaded."""
        return self._model is not None

    def __repr__(self) -> str:
        status = self._backend or "not loaded"
        return f"LLMBackend(model_id={self.model_id!r}, status={status})"


# ---------------------------------------------------------------------------
# Synthesis
# ---------------------------------------------------------------------------


def synthesize_answer(
    query: str,
    evidence_table: list[dict[str, Any]],
    graph_summary: dict[str, Any],
    confidence: dict[str, Any],
    llm: LLMBackend,
    expansions: list[dict[str, Any]] | None = None,
    lit_candidates: list[Any] | None = None,
) -> str:
    """Use the LLM to synthesise a natural-language answer from evidence.

    Parameters
    ----------
    query : str
        The original user question.
    evidence_table : list[dict]
        Provenance table produced by ``BioKGAgent._build_provenance_table``.
    graph_summary : dict
        Knowledge-graph summary from ``BioKnowledgeGraph.summary()``.
    confidence : dict
        Confidence scores from ``BioKGAgent._compute_confidence``.
    llm : LLMBackend
        A (possibly not-yet-loaded) LLM backend.

    Returns
    -------
    str
        Synthesised answer text.
    """
    # ── Build a structured evidence prompt from expansion data ─────────────
    # Priority: use rich expansion data if available (has drugs, interactions,
    # pathways, GO terms per gene). Fall back to raw evidence_table otherwise.

    sections: list[str] = []

    if expansions:
        for exp in expansions[:3]:
            gene = exp.get("gene", "")
            if not gene:
                continue
            gene_section: list[str] = [f"### Gene: {gene}"]

            # Approved drugs — name + mechanism only, no IDs
            approved = [d for d in exp.get("drugs", [])
                        if "approved" in [s.lower() for s in d.get("status", [])]]
            clinical = [d for d in exp.get("drugs", [])
                        if "approved" not in [s.lower() for s in d.get("status", [])]]
            if approved:
                gene_section.append("**FDA-Approved Targeted Drugs:**")
                for d in approved[:4]:
                    gene_section.append(
                        f"  - {d['drug_name']}: {d.get('mechanism', 'targeted therapy')}"
                    )
            if clinical and not approved:
                gene_section.append("**Clinical-Stage Compounds:**")
                for d in clinical[:3]:
                    st = ", ".join(d.get("status", ["investigational"]))
                    gene_section.append(f"  - {d['drug_name']} ({st}): {d.get('mechanism', '')}")

            # Top protein interactions — partners ONLY, no numeric scores
            interactions = sorted(exp.get("interactions", []),
                                  key=lambda x: -int(x.get("score", 0)))[:5]
            if interactions:
                partners = [iact["partner"] for iact in interactions if iact.get("partner")]
                gene_section.append(
                    f"**Top protein partners (high-confidence PPIs):** "
                    f"{', '.join(partners)}"
                )

            # Pathways — names only
            pathways = exp.get("pathways", [])[:4]
            if pathways:
                pw_names = [p.get("name", "") for p in pathways if p.get("name")]
                if pw_names:
                    gene_section.append(f"**Pathways:** {'; '.join(pw_names[:3])}")

            # GO biological process terms
            bp_terms = [t for t in exp.get("go_terms", [])
                        if t.get("namespace") == "biological_process"][:3]
            if bp_terms:
                go_names = [t.get("name", "") for t in bp_terms if t.get("name")]
                gene_section.append(f"**Biological processes:** {'; '.join(go_names[:3])}")

            sections.append("\n".join(gene_section))

    # Literature evidence — title + brief abstract snippet, no PMIDs
    lit_lines: list[str] = []
    if lit_candidates:
        for c in lit_candidates[:5]:
            payload = c.payload if hasattr(c, "payload") else c
            title = payload.get("title", "")
            abstract = payload.get("abstract", "")[:120]
            if title:
                line = f"  - \"{title}\""
                if abstract:
                    line += f" — {abstract}..."
                lit_lines.append(line)
    else:
        for item in evidence_table[:5]:
            if item.get("source_type") in ("dense", "bm25", "hybrid", "graph_retrieval"):
                title = item.get("title", "")
                if title:
                    lit_lines.append(f"  - \"{title}\"")

    if lit_lines:
        sections.append("### PubMed Literature Evidence\n" + "\n".join(lit_lines))

    evidence_block = "\n\n".join(sections) if sections else "No gene-specific evidence available."

    prompt = (
        f"## Biological Question\n{query}\n\n"
        f"## Evidence Retrieved from Databases\n{evidence_block}\n\n"
        f"## How to Answer\n"
        f"Write a clear, expert biological answer in 2-3 short prose paragraphs "
        f"(no bullet lists, no score numbers, no IDs):\n\n"
        f"Paragraph 1 — Directly answer the question in plain language. "
        f"If the question is about a disease (e.g. ulcer, diabetes), explain the "
        f"molecular targets that drive it and what the current standard treatments are.\n\n"
        f"Paragraph 2 — If approved drugs are in the evidence, name each one and explain "
        f"its mechanism of action. If there are no approved drugs in the evidence, say so "
        f"honestly and describe the closest research-stage compounds.\n\n"
        f"Paragraph 3 — Briefly mention 1-2 key protein interactions, the most relevant "
        f"biological pathway, and name 1-2 literature titles that support the answer.\n\n"
        f"Total response: under 300 words. Do NOT include confidence %, STRING scores, "
        f"DrugBank IDs, or PMIDs — the UI shows those separately."
    )

    return llm.generate(prompt, system_prompt=BIOKG_SYSTEM_PROMPT)


# ---------------------------------------------------------------------------
# Planner
# ---------------------------------------------------------------------------


def llm_planner(
    query: str,
    gene_catalog: list[str],
    llm: LLMBackend,
) -> dict[str, Any]:
    """LLM-powered query planning compatible with ``QueryRouter.plan()``.

    Parameters
    ----------
    query : str
        The user question.
    gene_catalog : list[str]
        Known gene symbols the system can resolve.
    llm : LLMBackend
        An LLM backend instance.

    Returns
    -------
    dict
        Keys compatible with ``QueryPlan`` construction in
        ``QueryRouter.plan()``:
        ``query_type``, ``retrieval_modes``, ``metadata_filters``,
        ``detected_entities``, ``use_reranker``,
        ``requires_graph_expansion``, ``requires_api_lookup``,
        ``max_iterations``, ``route_confidence``, ``rationale``.
    """
    # Provide a manageable subset of the gene catalog so the prompt stays short
    catalog_sample = gene_catalog[:200]
    catalog_str = ", ".join(catalog_sample)
    if len(gene_catalog) > 200:
        catalog_str += f" ... ({len(gene_catalog)} total)"

    prompt = (
        f"Known gene symbols: [{catalog_str}]\n\n"
        f"User question: {query}\n\n"
        f"Return the query plan as a JSON object."
    )

    raw = llm.generate(prompt, system_prompt=BIOKG_PLANNER_SYSTEM_PROMPT)

    # -- Parse JSON from LLM output -------------------------------------------
    plan = _extract_json(raw)

    # -- Validate and apply defaults ------------------------------------------
    valid_types = {"mechanistic", "relationship", "literature", "entity", "hybrid"}
    if plan.get("query_type") not in valid_types:
        plan["query_type"] = "hybrid"

    if not isinstance(plan.get("retrieval_modes"), list) or not plan["retrieval_modes"]:
        plan["retrieval_modes"] = ["dense", "bm25"]

    if not isinstance(plan.get("detected_entities"), list):
        plan["detected_entities"] = []

    if not isinstance(plan.get("metadata_filters"), dict):
        plan["metadata_filters"] = {"genes": plan.get("detected_entities", [])}

    plan.setdefault("use_reranker", True)
    plan.setdefault("requires_graph_expansion", True)
    plan.setdefault("requires_api_lookup", True)
    plan.setdefault("max_iterations", 2)
    plan.setdefault("route_confidence", 0.8)
    plan.setdefault("rationale", "LLM-planned route.")

    # Clamp route_confidence
    try:
        plan["route_confidence"] = max(0.0, min(1.0, float(plan["route_confidence"])))
    except (TypeError, ValueError):
        plan["route_confidence"] = 0.8

    try:
        plan["max_iterations"] = max(1, min(3, int(plan["max_iterations"])))
    except (TypeError, ValueError):
        plan["max_iterations"] = 2

    return plan


def _extract_json(text: str) -> dict[str, Any]:
    """Best-effort extraction of a JSON object from possibly noisy LLM output."""
    # Try direct parse first
    text = text.strip()
    try:
        return json.loads(text)  # type: ignore[no-any-return]
    except json.JSONDecodeError:
        pass

    # Strip markdown fences if present
    cleaned = re.sub(r"```(?:json)?\s*", "", text)
    cleaned = re.sub(r"```\s*$", "", cleaned).strip()
    try:
        return json.loads(cleaned)  # type: ignore[no-any-return]
    except json.JSONDecodeError:
        pass

    # Find first { ... } block
    match = re.search(r"\{[\s\S]*\}", text)
    if match:
        try:
            return json.loads(match.group())  # type: ignore[no-any-return]
        except json.JSONDecodeError:
            pass

    logger.warning("Could not parse JSON from LLM planner output; returning empty dict.")
    return {}


# ---------------------------------------------------------------------------
# Factory helpers
# ---------------------------------------------------------------------------


def create_llm_backend(
    model_id: str | None = None,
    device: str = DEFAULT_DEVICE,
    groq_api_key: str | None = None,
    groq_model: str | None = None,
    backend: str = "auto",
) -> GroqBackend | LLMBackend:
    """Create and return an LLM backend.

    Selection logic (when ``backend="auto"``):
    1. If ``GROQ_API_KEY`` env var or ``groq_api_key`` param is set → GroqBackend
    2. Otherwise → local LLMBackend (Kaggle T4 GPU)

    Parameters
    ----------
    model_id : str | None
        HuggingFace model identifier for local backend.
    device : str
        Device map string for local backend.
    groq_api_key : str | None
        Groq API key. Falls back to ``GROQ_API_KEY`` env var.
    groq_model : str | None
        Groq model name. Falls back to ``GROQ_MODEL`` env var.
    backend : str
        ``"auto"``, ``"groq"``, or ``"local"``.

    Returns
    -------
    GroqBackend | LLMBackend
    """
    api_key = groq_api_key or os.environ.get("GROQ_API_KEY", "")
    model = groq_model or os.environ.get("GROQ_MODEL", "llama-3.3-70b-versatile")

    if backend == "groq" or (backend == "auto" and api_key):
        logger.info("Using Groq API backend (model=%s)", model)
        return GroqBackend(api_key=api_key, model=model, max_tokens=DEFAULT_MAX_TOKENS)

    logger.info("Using local GPU backend (model=%s)", model_id or DEFAULT_MODEL_ID)
    return LLMBackend(
        model_id=model_id or DEFAULT_MODEL_ID,
        device=device,
        max_new_tokens=DEFAULT_MAX_TOKENS,
    )


def make_planner(llm: LLMBackend):
    """Return a callable ``(query, genes) -> dict`` for ``BioKGAgent.planner``.

    Usage::

        llm = create_llm_backend()
        agent = BioKGAgent.build(planner=make_planner(llm))

    Parameters
    ----------
    llm : LLMBackend
        An LLM backend instance (will be lazily loaded on first call).

    Returns
    -------
    Callable[[str, list[str]], dict[str, Any]]
    """

    def planner(query: str, genes: list[str]) -> dict[str, Any]:
        return llm_planner(query, genes, llm)

    return planner


# ---------------------------------------------------------------------------
# CLI smoke test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    print("=" * 60)
    print("BioKG-Agent LLM backend smoke test")
    print("=" * 60)

    backend = create_llm_backend()
    print(f"Backend created: {backend}")

    try:
        backend.load()
        print(f"Backend loaded: {backend}")

        test_prompt = (
            "What is the role of TP53 in cancer and how does it interact "
            "with MDM2?"
        )
        print(f"\nTest prompt: {test_prompt}")
        answer = backend.generate(test_prompt, system_prompt=BIOKG_SYSTEM_PROMPT)
        print(f"\nGenerated answer:\n{answer}")

        # Test planner
        print("\n" + "-" * 60)
        print("Testing LLM planner...")
        plan = llm_planner(
            test_prompt,
            ["TP53", "MDM2", "BRCA1", "EGFR", "KRAS"],
            backend,
        )
        print(f"Plan: {json.dumps(plan, indent=2)}")

    except RuntimeError as exc:
        print(f"\nCould not load model (expected in CPU-only env): {exc}")
        print("The module is importable and the API is functional.")
