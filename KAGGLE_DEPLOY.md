# BioKG-Agent: Kaggle Deployment Guide

A step-by-step guide to running BioKG-Agent on a free Kaggle T4 GPU using **Qwen2.5-7B-Instruct-AWQ** (no Claude API key required).

---

## Table of Contents

1. [Prerequisites](#1-prerequisites)
2. [Step-by-Step Kaggle Notebook Setup](#2-step-by-step-kaggle-notebook-setup)
3. [VRAM Budget](#3-vram-budget)
4. [Session Management Tips](#4-session-management-tips)
5. [Troubleshooting](#5-troubleshooting)
6. [Running Without GPU (CPU-Only Mode)](#6-running-without-gpu-cpu-only-mode)

---

## 1. Prerequisites

Before you begin, make sure you have the following:

| Requirement | Where to Get It | Cost |
|---|---|---|
| **Kaggle account** with GPU (T4) enabled | [kaggle.com](https://www.kaggle.com/) | Free |
| **NCBI API key** | [ncbi.nlm.nih.gov/account/settings](https://www.ncbi.nlm.nih.gov/account/settings/) | Free |
| **Ngrok auth token** | [dashboard.ngrok.com/get-started/your-authtoken](https://dashboard.ngrok.com/get-started/your-authtoken) | Free tier |
| **Groq API key** (optional) | [console.groq.com/keys](https://console.groq.com/keys) | Free tier (very generous) |

> **Groq vs Local GPU:** You have two LLM options. **Groq API** (free, no GPU needed, Llama 3.3 70B) or **Local Qwen2.5-7B** on Kaggle T4. Groq is easier and gives you a bigger model for free.

**How to enable GPU on Kaggle:**
1. Go to your Kaggle notebook settings (right sidebar).
2. Under **Accelerator**, select **GPU T4 x2** (or **GPU T4 x1**).
3. Set **Persistence** to **Files only** so outputs survive kernel restarts.

> **Note:** You do NOT need a Claude API key. This guide uses the open-source Qwen2.5-7B-Instruct-AWQ model running locally on the Kaggle GPU.

---

## 2. Step-by-Step Kaggle Notebook Setup

Create a new Kaggle notebook, enable GPU T4, and run the following cells in order.

### Cell 1: Install Dependencies

```python
!pip install -q requests networkx pyvis gradio numpy sentence-transformers faiss-cpu pyngrok torch transformers autoawq accelerate
```

> This takes 3-5 minutes on a fresh kernel. The `-q` flag keeps output minimal.

### Cell 2: Clone or Upload the Project

**Option A -- Upload as a Kaggle Dataset (recommended for persistence):**
1. Zip your `project1_biokg_agent` folder.
2. Go to [kaggle.com/datasets](https://www.kaggle.com/datasets) and create a new dataset.
3. Upload the zip file and publish.
4. In your notebook, add the dataset from the right sidebar under "Add data".

```python
# If uploaded as Kaggle dataset:
import shutil, os
shutil.copytree("/kaggle/input/YOUR_DATASET_NAME/project1_biokg_agent",
                "/kaggle/working/project1_biokg_agent")
os.chdir("/kaggle/working/project1_biokg_agent")
```

**Option B -- Clone from GitHub:**

```python
!git clone https://github.com/YOUR_USERNAME/project1_biokg_agent.git
%cd project1_biokg_agent
```

### Cell 3: Set Environment Variables

**Option A — Groq API (recommended, no GPU needed):**
```python
import os

os.environ["GROQ_API_KEY"]           = "YOUR_GROQ_API_KEY"  # Replace with your key
os.environ["GROQ_MODEL"]             = "llama-3.3-70b-versatile"
os.environ["BIOKG_LLM_BACKEND"]      = "groq"
os.environ["NGROK_AUTH_TOKEN"]       = "YOUR_NGROK_AUTH_TOKEN"  # Replace
os.environ["NCBI_API_KEY"]           = "YOUR_NCBI_KEY"       # Optional
os.environ["BIOKG_ENABLE_LIVE_APIS"] = "1"
```

**Option B — Local Qwen on T4 GPU:**
```python
import os

os.environ["BIOKG_MODEL_ID"]         = "Qwen/Qwen2.5-7B-Instruct-AWQ"
os.environ["BIOKG_DEVICE"]           = "cuda"
os.environ["BIOKG_LLM_BACKEND"]      = "local"
os.environ["NGROK_AUTH_TOKEN"]       = "YOUR_NGROK_AUTH_TOKEN"  # Replace
os.environ["NCBI_API_KEY"]           = "YOUR_NCBI_KEY"       # Optional
os.environ["BIOKG_ENABLE_LIVE_APIS"] = "1"
```

### Cell 4: Verify GPU

```python
import torch

assert torch.cuda.is_available(), "GPU not detected! Check notebook settings -> Accelerator -> GPU T4"

print(f"GPU available : {torch.cuda.is_available()}")
print(f"GPU name      : {torch.cuda.get_device_name(0)}")
print(f"VRAM          : {torch.cuda.get_device_properties(0).total_mem / 1e9:.1f} GB")
```

Expected output:
```
GPU available : True
GPU name      : Tesla T4
VRAM          : 15.8 GB
```

### Cell 5: Test LLM Loading

```python
from biokg_agent.llm import create_llm_backend

llm = create_llm_backend()
llm.load()

# Quick sanity check
response = llm.generate("What is TP53?", system_prompt="You are a biology expert.")
print(response)
```

> First run downloads the model (~4 GB). Subsequent runs use the Kaggle cache. Loading takes about 30-60 seconds on T4.

### Cell 6: Build the Agent with LLM

```python
from biokg_agent.agent import BioKGAgent
from biokg_agent.config import ProjectConfig

config = ProjectConfig.from_env()
agent = BioKGAgent.build(config=config)
agent.attach_llm(llm)

print("Agent ready!")
```

### Cell 7: Test a Query

```python
result = agent.invoke("What drugs target TP53 pathway proteins?")
print(result["answer_text"])
```

### Cell 8: Run Smoke Eval

```python
from run_demo import run_smoke_eval

report = run_smoke_eval(agent)
print(f"Pass rate: {report['pass_rate']:.0%}")
```

### Cell 9: Run Benchmark (Optional)

```python
import json

with open("eval/benchmark.json") as f:
    benchmark = json.load(f)

# Run first 5 questions as a quick test
for q in benchmark["questions"][:5]:
    result = agent.invoke(q["question"])
    print(f"Q: {q['question'][:60]}...")
    print(f"A: {result['answer_text'][:100]}...")
    print()
```

### Cell 10: Launch Gradio UI with Ngrok Tunnel

```python
from pyngrok import ngrok

# Authenticate ngrok
ngrok.set_auth_token(os.environ["NGROK_AUTH_TOKEN"])

# Build and launch the Gradio app
from biokg_agent.app import build_app

demo = build_app()
tunnel = ngrok.connect(7860)
print(f"\n{'='*50}")
print(f"  Public URL: {tunnel.public_url}")
print(f"{'='*50}\n")
print("Share this URL with anyone to access the BioKG-Agent UI.")
print("The tunnel stays active as long as this cell is running.\n")

demo.launch(server_port=7860, share=False)
```

> Open the printed URL in your browser to access the full Gradio interface.

---

## 3. VRAM Budget

The Kaggle T4 GPU has **15.8 GB** of VRAM. Here is how BioKG-Agent uses it:

| Component | VRAM Usage | Notes |
|---|---|---|
| Qwen2.5-7B-Instruct-AWQ (4-bit) | ~5.0 GB | Main LLM, quantized to 4-bit |
| Sentence-transformers (MiniLM) | ~0.3 GB | Embedding model for RAG |
| Cross-encoder reranker | ~0.3 GB | Re-ranks retrieved passages |
| FAISS index | 0 GB | Runs on CPU RAM |
| NetworkX graph | 0 GB | Runs on CPU RAM |
| PyTorch overhead / CUDA kernels | ~0.5 GB | Runtime overhead |
| **Total** | **~6.1 GB / 15.8 GB** | **~10 GB headroom** |

You have plenty of room. The large headroom means inference will not trigger OOM errors even with long context queries.

---

## 4. Session Management Tips

Kaggle GPU sessions have limits. Plan accordingly:

| Limit | Value |
|---|---|
| Max session duration | 12 hours |
| Weekly GPU quota | 30 hours |
| Persistent storage | `/kaggle/working/` (within session) |

**Best practices:**

- **Save checkpoints frequently.** Before your session ends or before running risky code:
  ```python
  agent.save("/kaggle/working/biokg_checkpoint")
  ```

- **Upload data as a Kaggle dataset** for persistence across sessions. Files in `/kaggle/working/` are lost when the session ends, but Kaggle datasets persist forever.

- **Monitor your GPU quota** from the Kaggle settings page. If you are running low, switch to CPU-only mode (see Section 6).

- **Download results before the session ends:**
  ```python
  # Save results to a file, then download from the Kaggle output tab
  import json
  with open("/kaggle/working/results.json", "w") as f:
      json.dump(results, f, indent=2)
  ```

- **Restart kernel if VRAM fills up.** Go to Run -> Restart & Clear All Outputs, then re-run cells from the top.

---

## 5. Troubleshooting

### GPU not detected

```
AssertionError: GPU not detected!
```

**Fix:** Open the notebook sidebar -> Settings -> Accelerator -> select **GPU T4 x2**. Then restart the kernel.

### Out of memory (OOM)

```
torch.cuda.OutOfMemoryError: CUDA out of memory
```

**Fix:**
1. Check current VRAM usage:
   ```python
   import torch
   print(f"Allocated: {torch.cuda.memory_allocated() / 1e9:.1f} GB")
   print(f"Reserved:  {torch.cuda.memory_reserved() / 1e9:.1f} GB")
   ```
2. Free unused memory:
   ```python
   torch.cuda.empty_cache()
   ```
3. If that does not help, restart the kernel and re-run all cells.

### Model download hangs or fails

```
ConnectionError / ReadTimeoutError
```

**Fix:** Kaggle has intermittent network issues. Simply re-run the cell. If it keeps failing, add a retry:
```python
import os
os.environ["HF_HUB_DOWNLOAD_TIMEOUT"] = "300"  # 5 minutes
```

### Ngrok tunnel not working

```
PyngrokNgrokError: ngrok returned an error
```

**Fix:**
1. Make sure your auth token is correct -- get it from [dashboard.ngrok.com](https://dashboard.ngrok.com/get-started/your-authtoken).
2. Free-tier ngrok allows only 1 tunnel at a time. Kill existing tunnels:
   ```python
   from pyngrok import ngrok
   ngrok.kill()
   ```
3. Re-run Cell 10.

### Import errors (module not found)

```
ModuleNotFoundError: No module named 'biokg_agent'
```

**Fix:** Make sure you are in the correct directory:
```python
import os
print(os.getcwd())  # Should end with 'project1_biokg_agent'
# If not:
os.chdir("/kaggle/working/project1_biokg_agent")
```

### NCBI API rate limits

If you see `HTTP 429 Too Many Requests` from NCBI APIs:
- Add your free NCBI API key (raises limit from 3 to 10 requests/second):
  ```python
  os.environ["NCBI_API_KEY"] = "your_actual_ncbi_key"
  ```
- Get one at: https://www.ncbi.nlm.nih.gov/account/settings/

---

## 6. Running Without GPU (CPU-Only Mode)

If you have exhausted your GPU quota or want to run on a CPU-only Kaggle kernel, the agent still works -- just without LLM-powered answer synthesis.

### Setup for CPU-only mode

Replace Cell 3 environment variables with:

```python
import os

os.environ["NCBI_API_KEY"]           = "YOUR_NCBI_KEY"
os.environ["BIOKG_DEVICE"]           = "cpu"
os.environ["BIOKG_ENABLE_LIVE_APIS"] = "1"
```

### Build the agent without LLM

```python
from biokg_agent.agent import BioKGAgent
from biokg_agent.config import ProjectConfig

config = ProjectConfig.from_env()
config.enable_llm_synthesis = False  # Disable LLM, use template-based answers

agent = BioKGAgent.build(config=config)
print("Agent ready (CPU-only, template-based answers)")
```

### What still works in CPU-only mode

| Feature | Available? |
|---|---|
| Knowledge graph queries (NetworkX) | Yes |
| NCBI/PubMed live API lookups | Yes |
| FAISS vector retrieval | Yes (slower) |
| Sentence-transformer embeddings | Yes (slower) |
| Cross-encoder reranking | Yes (slower) |
| LLM answer synthesis | No (uses templates instead) |
| Gradio UI | Yes |

Template-based answers combine retrieved facts into structured bullet-point responses. They are less fluent than LLM-generated answers but still accurate and useful.

---

## Quick Reference: Copy-Paste All Cells

For convenience, here is a single cell you can paste to get the full agent running (after installing dependencies):

```python
# === QUICK START (run after Cell 1 installs) ===
import os, torch

# Config
os.environ["NGROK_AUTH_TOKEN"]       = "YOUR_NGROK_TOKEN"
os.environ["NCBI_API_KEY"]           = "YOUR_NCBI_KEY"
os.environ["BIOKG_MODEL_ID"]         = "Qwen/Qwen2.5-7B-Instruct-AWQ"
os.environ["BIOKG_DEVICE"]           = "cuda"
os.environ["BIOKG_ENABLE_LIVE_APIS"] = "1"

# Verify GPU
assert torch.cuda.is_available(), "Enable GPU in notebook settings!"

# Load LLM
from biokg_agent.llm import create_llm_backend
llm = create_llm_backend()
llm.load()

# Build agent
from biokg_agent.agent import BioKGAgent
from biokg_agent.config import ProjectConfig
config = ProjectConfig.from_env()
agent = BioKGAgent.build(config=config)
agent.attach_llm(llm)

# Test
result = agent.invoke("What drugs target TP53 pathway proteins?")
print(result["answer_text"])

# Launch UI
from pyngrok import ngrok
from biokg_agent.app import build_app
ngrok.set_auth_token(os.environ["NGROK_AUTH_TOKEN"])
demo = build_app()
tunnel = ngrok.connect(7860)
print(f"Public URL: {tunnel.public_url}")
demo.launch(server_port=7860, share=False)
```

---

*Last updated: 2026-04-06*
