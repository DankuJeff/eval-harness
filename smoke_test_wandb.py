"""
Step 9 smoke test — Haiku + Sonnet, general mode, v1 prompts, W&B logging.
Verifies runs appear in the W&B project dashboard with correct metric fields.

Run: python smoke_test_wandb.py
Then open wandb.ai → eval-harness project and confirm both runs are visible.
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from dotenv import load_dotenv
from runners.model_runner import ModelRunner
from tracking.wandb_logger import WandbLogger

load_dotenv()

DATASET_PATH = Path(__file__).parent / "datasets" / "code_eval_dataset.json"
SMOKE_CASE_LIMIT = 2


def main() -> None:
    with open(DATASET_PATH) as f:
        dataset = json.load(f)

    non_rag = [c for c in dataset if c.get("rag_context") is None]
    smoke_dataset = non_rag[:SMOKE_CASE_LIMIT]

    for model_name in ("haiku", "sonnet"):
        print(f"\nRunning {model_name.upper()} — logging to W&B...")
        logger = WandbLogger(run_name=f"smoke-{model_name}-v1-general")

        runner = ModelRunner(model=model_name, prompt_version="v1")
        results = runner.run_general(smoke_dataset)

        logger.log_run(results)
        logger.finish()

        print(f"  Logged {len(results)} results for {model_name}.")
        for r in results:
            print(f"    [{r['case_id']}] pass_rate={r['pass_rate']:.2f} | "
                  f"metrics={list(r['metric_scores'].keys())}")

    print("\nW&B smoke test complete.")
    print("Open wandb.ai → eval-harness project and verify both runs appear.\n")


if __name__ == "__main__":
    main()
