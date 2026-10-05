# Gen-AI via LangChain

The repository tracks the course end to end: LangChain fundamentals, through RAG systems, to end-to-end AI agents.

Everything here is written while working through the material, so each folder contains runnable scripts
rather than snippets, and the notes PDFs are consolidated per topic rather than per video.

---

## Progress

| Lectures | Topic | Status | Folder / notes |
|---|---|---|---|
| 1–3 | Roadmap, Introduction to LangChain, the six components | Done | [notes PDF](01-03_LangChain_Fundamentals_Notes.pdf) |
| 4–5 | Models — chat models, open-source models, embeddings | Done | `Langchain and Rag/` · [notes PDF](04-05_LangChain_Models_Notes.pdf) |
| 6 | Prompts — `PromptTemplate`, `ChatPromptTemplate`, messages, `MessagesPlaceholder` | Done | `Langchain_Prompts/` |
| 7 | Structured output — `with_structured_output` with TypedDict, Pydantic, JSON Schema | Done | `Structured_Outputs/` · [notes PDF](Structured_Outputs/GenAI-Notes-Structured-Output-and-Parsers.pdf) |
| 8 | Output parsers — Str, JSON, Structured, Pydantic | Done | `Output_parsers/` · same notes PDF as 7 |
| 9 | Chains — sequential, parallel, conditional | Done | `chains/` · [notes PDF](chains/LangChain_Chains_Notes.pdf) |
| 10+ | Runnables & LCEL, document loaders, text splitters, vector stores, retrievers, RAG, agents | Next | — |

A full set of running course notes is also at the root: [`Langchain notes.pdf`](<Langchain notes.pdf>).

---

## Repository structure

```
Gen-AI-via-Langchain/
├── 01-03_LangChain_Fundamentals_Notes.pdf
├── 04-05_LangChain_Models_Notes.pdf
├── Langchain notes.pdf                     running course notes
│
├── Langchain and Rag/                      Models (lectures 4–5)
│   ├── requirements.txt
│   ├── Langchain_models.ipynb              environment checks, package versions
│   ├── LLM/
│   │   ├── llm_demo.py                     minimal reusable wrapper
│   │   └── llm_models.ipynb
│   ├── ChatModels/
│   │   ├── chatmodel_openai_1.py           basic ChatOpenAI call
│   │   ├── chatmodel_openai_2.py           temperature sweep, 0 → 1.7
│   │   ├── 4_chatmodel_hf_api.py           Hugging Face Inference API
│   │   ├── 5_chatmodel_hf_local.py         local weights via HuggingFacePipeline
│   │   ├── 6_chatmodel_hf_local_ollama.py  local via Ollama
│   │   └── chatmodel.ipynb
│   └── Embedded_models/
│       ├── embedding_openai_query.py       embed a single string
│       ├── embedding_openai_docs.py        embed a list of documents
│       ├── embedding_hf_lcoal.py           local embeddings, all-MiniLM-L6-v2
│       ├── embedding_hf_docs.py
│       ├── Document_similarity.py          cosine similarity retrieval
│       └── embedding_model.ipynb
│
├── Langchain_Prompts/                      Prompts (lecture 6)
│   ├── prompt_generator.py                 builds a PromptTemplate, saves it to template.json
│   ├── template.json                       the saved, reusable template
│   ├── prompt_UI.py                        Streamlit research-paper explainer, loads template.json
│   ├── Messages.py                         System / Human / AI messages
│   ├── chatbot.py                          CLI chatbot that keeps chat history
│   ├── chat_prompt_template.py             ChatPromptTemplate with variables
│   ├── Message_Placeholder.py              MessagesPlaceholder + stored history
│   └── chat_history.txt                    sample history for the placeholder demo
│
├── Structured_Outputs/                     Structured output (lecture 7)
│   ├── structured_output.py                TypedDict schema, two fields
│   ├── Complex_structured_output.py        TypedDict with Annotated, Literal, Optional
│   ├── pydantic_structured_output.py       same review schema as a Pydantic model
│   ├── json_schema_structured_output.py    same schema as raw JSON Schema
│   ├── TypedDict.ipynb                     TypedDict basics
│   └── GenAI-Notes-Structured-Output-and-Parsers.pdf
│
├── Output_parsers/                         Output parsers (lecture 8)
│   ├── strputputparsers.py                 two-step report → summary, done manually
│   ├── stroutputparser1.py                 same pipeline as one chain with StrOutputParser
│   ├── jsonoutputparsers.py                JsonOutputParser
│   ├── Structured_output_parser.py         StructuredOutputParser + ResponseSchema
│   ├── pydanticoutputparser.py             PydanticOutputParser with field validation
│   ├── langchain_classic.ipynb             installing langchain-classic
│   └── GenAI-Notes-Structured-Output-and-Parsers.pdf
│
└── chains/                                 Chains (lecture 9)
    ├── simple_chain.py                     prompt | model | parser
    ├── sequential_chain.py                 report → 5-point summary
    ├── parallel_chain.py                   notes + quiz in parallel, then merged
    ├── conditional_chains.py               sentiment classifier → RunnableBranch
    ├── chains.ipynb
    └── LangChain_Chains_Notes.pdf
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

Create a `.env` file at the **repository root**, so every folder finds it (`load_dotenv()` searches
upward from the script's folder):

```
OPENAI_API_KEY=sk-proj-...
HF_TOKEN=hf_...
```

No quotes, no trailing spaces — a single trailing space produces an `Illegal header value` error
that surfaces as a misleading "Connection error". The scripts also call `.strip()` on the key as a
safety net. `.env` is listed in `.gitignore`; never commit it.

**3. Optional, per topic**

| To run | Install / do |
|---|---|
| `5_chatmodel_hf_local.py` | `pip install accelerate` (only if using `device_map`) |
| `embedding_hf_*.py` | `pip install sentence-transformers` |
| `6_chatmodel_hf_local_ollama.py` | [Install Ollama](https://ollama.com), then `ollama pull llama3.1` |
| Gated models (Llama) | Accept the licence on the model's Hugging Face page, then `huggingface_hub.login()` |
| `prompt_UI.py` | `pip install streamlit` |
| `Structured_output_parser.py` | `pip install langchain-classic` (LangChain 1.x moved `StructuredOutputParser` there) |
| `chain.get_graph().print_ascii()` | `pip install grandalf` |

**4. Model cache location (optional)**

Hugging Face downloads default to `C:\Users\<you>\.cache\huggingface` and grow quickly. To relocate,
set `HF_HOME` as a **system** environment variable rather than in Python — the library reads the path
once at import, so `os.environ[...]` set inside a script is often too late.

---

## Running the scripts

Each file is standalone. Run it from inside its own folder, since some scripts read files by relative
path (`template.json`, `chat_history.txt`):

```bash
cd chains
python parallel_chain.py
```

The Streamlit app needs its own runner:

```bash
cd Langchain_Prompts
streamlit run prompt_UI.py
```

From a notebook:

```python
%run chatmodel_openai_1.py
```

---

## What each section covers

**Models (4–5).** The four chat-model scripts call four different backends — a paid API, a hosted
open-source endpoint, local GPU inference, and a local Ollama server. The last three lines of each are
identical, which is the entire argument for the abstraction:

```python
model = ChatOpenAI(model="gpt-4", ...)          # or ChatHuggingFace, or ChatOllama
result = model.invoke("...")
print(result.content)
```

The embedding scripts build toward the same idea from the retrieval side, ending at
`Document_similarity.py` — a working semantic search over five documents, which is RAG with the
generation step removed. Full write-up in [`04-05_LangChain_Models_Notes.pdf`](04-05_LangChain_Models_Notes.pdf),
including a field-notes appendix of the errors hit along the way (gated repos, VRAM limits, greedy
decoding silently ignoring `temperature`).

**Prompts (6).** Static vs dynamic prompts: a `PromptTemplate` is built once, saved to `template.json`
and loaded by a Streamlit UI. The message scripts show why chat history matters — `chatbot.py` keeps
a running list of `SystemMessage` / `HumanMessage` / `AIMessage`, and `MessagesPlaceholder` injects
stored history into a `ChatPromptTemplate`.

**Structured output (7).** Getting dicts or objects out of a model instead of free text, using
`model.with_structured_output(schema)`. The same product-review schema is written three ways —
TypedDict (type hints only), Pydantic (validated at runtime), and JSON Schema (language-agnostic) — so
the trade-offs are easy to compare.

**Output parsers (8).** For models without native structured output (many Hugging Face models), the
parser does the work instead:

| Parser | Returns | Validates types? |
|---|---|---|
| `StrOutputParser` | `str` | — |
| `JsonOutputParser` | `dict` | No, keys not enforced without a schema |
| `StructuredOutputParser` | `dict` of strings | No |
| `PydanticOutputParser` | Pydantic object | Yes |

**Chains (9).** LCEL pipelines with the `|` operator: a simple chain, a sequential chain that feeds one
model's answer into a second prompt, a `RunnableParallel` chain that writes notes and a quiz at the same
time (one branch on OpenAI, one on Hugging Face) and merges them, and a conditional chain that classifies
feedback with a Pydantic parser and routes it with `RunnableBranch`. `RunnablePassthrough.assign` keeps
the original input alongside the classification. The notes PDF ends with a cheat sheet and a list of
the bugs hit while building these.

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
