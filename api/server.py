"""
Step 10 — FastAPI server.

Exposes POST /run — accepts mode, model(s), and prompt_versions.
Calls the same ModelRunner / PromptRunner logic as the CLI.
Both entry points produce identical W&B log entries for identical parameters.

Start: uvicorn api.server:app --reload
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, field_validator

from runners.model_runner import MODEL_MAP, ModelRunner
from runners.prompt_runner import PromptRunner
from runners.retriever import get_retriever
from tracking.wandb_logger import WandbLogger

load_dotenv()

app = FastAPI(title="Eval Harness API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],
    allow_methods=["GET", "POST"],
    allow_headers=["Content-Type"],
)

DATASET_PATH = Path(__file__).parent.parent / "datasets" / "code_eval_dataset.json"

with open(DATASET_PATH) as f:
    DATASET = json.load(f)


class RunRequest(BaseModel):
    mode: str
    model: list[str]
    prompt_versions: list[str] = ["v1"]

    @field_validator("mode")
    @classmethod
    def validate_mode(cls, v: str) -> str:
        if v not in ("general", "rag"):
            raise ValueError("mode must be 'general' or 'rag'")
        return v

    @field_validator("model")
    @classmethod
    def validate_models(cls, v: list[str]) -> list[str]:
        invalid = [m for m in v if m not in MODEL_MAP]
        if invalid:
            raise ValueError(f"Unknown model(s): {invalid}. Valid: {list(MODEL_MAP.keys())}")
        return v

    @field_validator("prompt_versions")
    @classmethod
    def validate_versions(cls, v: list[str]) -> list[str]:
        invalid = [p for p in v if p not in ("v1", "v2", "v3")]
        if invalid:
            raise ValueError(f"Unknown prompt version(s): {invalid}. Valid: v1, v2, v3")
        return v


@app.post("/run")
def run_eval(req: RunRequest) -> dict:
    retriever = get_retriever() if req.mode == "rag" else None
    all_results: list[dict] = []

    logger = WandbLogger(
        run_name=f"api-{'-'.join(req.model)}-{'_'.join(req.prompt_versions)}-{req.mode}"
    )

    try:
        for model_name in req.model:
            if len(req.prompt_versions) == 1:
                runner = ModelRunner(model=model_name, prompt_version=req.prompt_versions[0])
                if req.mode == "general":
                    results = runner.run_general(DATASET)
                else:
                    results = runner.run_rag(DATASET, retriever)
                all_results.extend(results)
            else:
                prompt_runner = PromptRunner(model=model_name, prompt_versions=req.prompt_versions)
                results_by_version = prompt_runner.run(DATASET, mode=req.mode)
                for version_results in results_by_version.values():
                    all_results.extend(version_results)

        logger.log_run(all_results)
    finally:
        logger.finish()

    total = sum(len(r["metric_pass"]) for r in all_results)
    passed = sum(ok for r in all_results for ok in r["metric_pass"].values())

    return {
        "results": all_results,
        "summary": {
            "total_cases": len(all_results),
            "total_metric_checks": total,
            "passed": passed,
            "pass_rate": round(passed / total, 3) if total > 0 else 0.0,
        },
    }


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}
