# LLM Eval Harness — Code Understanding & Review

A production-grade evaluation harness that tests large language models on code understanding and review tasks. The harness runs structured eval suites across multiple models and prompt versions, tracks all experiment results in Weights & Biases, and surfaces regression data in a React dashboard. Both a CLI and a dashboard Run button serve as entry points to the same underlying runner logic.

Supported task types: bug identification, code explanation, code review, and security issue flagging. Two evaluation modes are available — general LLM mode (prompt in, response evaluated directly) and RAG mode (retrieved code context injected alongside the prompt, evaluated on both retrieval quality and generation quality). Models under test: Claude Haiku, Sonnet, and Opus via the Anthropic API. Scoring uses DeepEval's LLM-as-judge metrics, run through DeepEval's default OpenAI judge — resolved from `OPENAI_API_KEY` and not pinned in code, so it can change as DeepEval tracks OpenAI releases (this project started on `gpt-4o` and DeepEval auto-upgraded to `gpt-5.4` mid-run).

---

## Prerequisites

- Python 3.11+
- Node.js 18+
- Anthropic API key
- OpenAI API key (required for DeepEval's judge model and RAG embeddings)
- Weights & Biases account and API key

---

## Setup

```bash
# Clone the repo and enter the project root
git clone <repo-url>
cd eval-harness

# Create and activate a Python virtual environment
python -m venv venv
venv\Scripts\activate        # Windows
source venv/bin/activate     # macOS / Linux

# Install Python dependencies
pip install -r requirements.txt

# Copy the environment template and fill in your API keys
cp .env.example .env
# Edit .env and set: ANTHROPIC_API_KEY, OPENAI_API_KEY, WANDB_API_KEY, WANDB_PROJECT, WANDB_ENTITY

# Install dashboard dependencies
cd dashboard
npm install
cd ..
```

---

## Usage

### CLI

```bash
# General mode — Haiku, prompt v1
python cli.py --mode general --model haiku --prompt-version v1

# RAG mode — Sonnet, prompt v2
python cli.py --mode rag --model sonnet --prompt-version v2

# Multi-model comparison
python cli.py --mode general --model haiku sonnet --prompt-version v1

# Prompt regression — all three versions logged to W&B for diff
python cli.py --mode general --model haiku --compare-prompts
```

**Flags:**

| Flag | Values | Default | Description |
|------|--------|---------|-------------|
| `--mode` | `general`, `rag` | required | Eval mode |
| `--model` | `haiku`, `sonnet`, `opus` | `haiku` | One or more models |
| `--prompt-version` | `v1`, `v2`, `v3` | `v1` | One or more prompt versions |
| `--compare-prompts` | — | off | Run all three prompt versions and log diffs to W&B |

### API Server

```bash
# Start the FastAPI server
uvicorn api.server:app --reload --port 8001

# Trigger a run via curl
curl.exe -X POST http://127.0.0.1:8001/run \
  -H "Content-Type: application/json" \
  -d "{\"mode\": \"general\", \"model\": [\"haiku\"], \"prompt_versions\": [\"v1\"]}"
```

### Dashboard

```bash
# With the API server running, start the dashboard
cd dashboard
npm run dev
# Open http://localhost:5173
```

---

## Architecture

```
eval-harness/
├── datasets/
│   └── code_eval_dataset.json   ← 32 ground-truth test cases (24 general + 8 RAG)
├── evals/
│   ├── general_llm_tests.py     ← standalone eval script for general mode
│   └── rag_tests.py             ← standalone eval script for RAG mode
├── metrics/
│   └── custom_metrics.py        ← metric definitions per task type
├── runners/
│   ├── model_runner.py          ← loops across models, returns structured results
│   ├── prompt_runner.py         ← loops across prompt versions, calls model_runner
│   └── retriever.py             ← in-memory vector store, OpenAI embeddings, cosine similarity
├── tracking/
│   └── wandb_logger.py          ← W&B integration, injected into runners at entry point
├── api/
│   └── server.py                ← FastAPI — POST /run, calls same runners as CLI
├── dashboard/
│   └── src/
│       ├── App.jsx              ← run state, layout
│       └── components/
│           ├── RunButton.jsx    ← mode/model/version controls, POSTs to /run
│           ├── ResultsTable.jsx ← per-metric results with pass/fail
│           ├── RegressionChart.jsx  ← avg scores by prompt version (Recharts)
│           └── ModelComparison.jsx  ← pass rate by model (Recharts)
├── cli.py                       ← CLI entry point — calls runners directly
├── test_retrieval.py            ← manual retrieval verification script
├── smoke_test_models.py         ← smoke test for multi-model runner
├── requirements.txt
└── .env.example
```

**Entry point unification:** `cli.py` and `api/server.py` are two interfaces to the same runner logic. Neither owns any eval logic — all evaluation runs through `runners/`.

**W&B logging:** `WandbLogger` is initialized once at the entry point and injected into runners as a dependency. This prevents multiple `wandb.init()` calls within a single execution.

**Pass thresholds:** 0.7 across all metrics except security flagging Correctness (0.8) and Hallucination (0.3 — inverted scale, lower is better).
