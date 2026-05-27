"""
Step 7 smoke test — Haiku + Sonnet, general mode, v1 prompts.
Verifies ModelRunner returns structured result objects for both models.
Opus is skipped to conserve API credits.

Run: python smoke_test_models.py
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from dotenv import load_dotenv
from runners.model_runner import ModelRunner

load_dotenv()

DATASET_PATH = Path(__file__).parent / "datasets" / "code_eval_dataset.json"

# Run only 2 cases per model to keep the smoke test fast and cheap
SMOKE_CASE_LIMIT = 2


def main() -> None:
    with open(DATASET_PATH) as f:
        dataset = json.load(f)

    # Limit to first N non-RAG cases for speed
    non_rag = [c for c in dataset if c.get("rag_context") is None]
    smoke_dataset = non_rag[:SMOKE_CASE_LIMIT]

    for model_name in ("haiku", "sonnet"):
        print(f"\n{'=' * 60}")
        print(f"Running smoke test: {model_name.upper()} — {SMOKE_CASE_LIMIT} cases, general mode, v1")
        print("=" * 60)

        runner = ModelRunner(model=model_name, prompt_version="v1")
        results = runner.run_general(smoke_dataset)

        print(f"\nReturned {len(results)} result objects.")
        for r in results:
            print(f"  [{r['case_id']}] model={r['model']} | prompt={r['prompt_version']} | "
                  f"mode={r['mode']} | pass_rate={r['pass_rate']:.2f}")
            for metric, score in r["metric_scores"].items():
                print(f"    {metric}: {score:.3f}")

    print("\nSmoke test complete. Both models returned structured results.\n")


if __name__ == "__main__":
    main()
