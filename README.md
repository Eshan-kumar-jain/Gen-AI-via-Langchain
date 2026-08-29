# Gen-AI via LangChain

 The repository
tracks the course end to end: LangChain fundamentals, through RAG systems, to end-to-end AI agents.

Everything here is written while working through the material, so each folder contains runnable scripts
rather than snippets, and the notes PDFs are consolidated per topic rather than per video.

---

## Progress

| Lectures | Topic | Status | Folder |
|---|---|---|---|
| 1–3 | Roadmap, Introduction to LangChain, the six components | Done | [notes PDF](01-03_LangChain_Fundamentals_Notes.pdf) |
| 4–5 | Models — chat models, open-source models, embeddings | Done | `ChatModels/`, `Embedded_models/` |
| 6 | Prompts | Next | — |
| 7+ | Chains, Indexes, Memory, Agents, RAG | Planned | — |

---

## Repository structure

```
Gen-AI-via-Langchain/
├── 01-03_LangChain_Fundamentals_Notes.pdf
├── 04-05_LangChain_Models_Notes.pdf
└── Langchain and Rag/
    ├── requirements.txt
    ├── Langchain_models.ipynb          environment checks, package versions
    ├── LLM/
    │   ├── llm_demo.py                 minimal reusable wrapper
    │   └── llm_models.ipynb
    ├── ChatModels/
    │   ├── chatmodel_openai_1.py       basic ChatOpenAI call
    │   ├── chatmodel_openai_2.py       temperature sweep, 0 → 1.7
    │   ├── 4_chatmodel_hf_api.py       Hugging Face Inference API
    │   ├── 5_chatmodel_hf_local.py     local weights via HuggingFacePipeline
    │   ├── 6_chatmodel_hf_local_ollama.py   local via Ollama
    │   └── chatmodel.ipynb
    └── Embedded_models/
        ├── embedding_openai_query.py   embed a single string
        ├── embedding_openai_docs.py    embed a list of documents
        ├── embedding_hf_lcoal.py       local embeddings, all-MiniLM-L6-v2
        ├── embedding_hf_docs.py
        ├── Document_similarity.py      cosine similarity retrieval
        └── embedding_model.ipynb
```

---

## Setup

**1. Environment**

```bash
conda create -n ml_env python=3.10
conda activate ml_env
pip install -r "Langchain and Rag/requirements.txt"
```

**2. API keys**

Create a `.env` file inside `Langchain and Rag/`:

```
OPENAI_API_KEY=sk-proj-...
HF_TOKEN=hf_...
```

No quotes, no trailing spaces — a single trailing space produces an `Illegal header value` error
that surfaces as a misleading "Connection error". `.env` is gitignored and must stay that way.

**3. Optional, per backend**

| To run | Install / do |
|---|---|
| `5_chatmodel_hf_local.py` | `pip install accelerate` (only if using `device_map`) |
| `embedding_hf_*.py` | `pip install sentence-transformers` |
| `6_chatmodel_hf_local_ollama.py` | [Install Ollama](https://ollama.com), then `ollama pull llama3.1` |
| Gated models (Llama) | Accept the licence on the model's Hugging Face page, then `huggingface_hub.login()` |

**4. Model cache location (optional)**

Hugging Face downloads default to `C:\Users\<you>\.cache\huggingface` and grow quickly. To relocate,
set `HF_HOME` as a **system** environment variable rather than in Python — the library reads the path
once at import, so `os.environ[...]` set inside a script is often too late.

---

## Running the scripts

Each file is standalone:

```bash
cd "Langchain and Rag/ChatModels"
python chatmodel_openai_1.py
```

Or from the notebooks, which is how they were developed:

```python
%run chatmodel_openai_1.py
```

---

## What the Models section covers

The four chat-model scripts call four different backends — a paid API, a hosted open-source endpoint,
local GPU inference, and a local Ollama server. The last three lines of each are identical, which is
the entire argument for the abstraction:

```python
model = ChatOpenAI(model="gpt-4", ...)          # or ChatHuggingFace, or ChatOllama
result = model.invoke("...")
print(result.content)
```

The embedding scripts build toward the same idea from the retrieval side, ending at
`Document_similarity.py` — a working semantic search over five documents, which is RAG with the
generation step removed.

Full write-up in [`04-05_LangChain_Models_Notes.pdf`](04-05_LangChain_Models_Notes.pdf), including a
field-notes appendix of the errors hit along the way (gated repos, VRAM limits, greedy decoding
silently ignoring `temperature`) and what each one turned out to be.

---

## Hardware note

Local inference was developed on a GTX 1650 (4 GB VRAM). Rough sizing:

| Model size | fp16 | 4-bit (Ollama) | Fits in 4 GB? |
|---|---|---|---|
| 1.5B | ~3 GB | ~1 GB | Yes |
| 3B | ~6 GB | ~2 GB | Quantised only |
| 7–8B | ~16 GB | ~4.7 GB | Quantised, with spill |
| 13B+ | ~26 GB | ~8 GB | No |

Exceeding VRAM usually degrades rather than fails — `device_map="auto"` moves the overflow to system
RAM and generation slows dramatically, which looks like a hang.

---
## Licence

Study notes and practice code. Free to use for learning.
