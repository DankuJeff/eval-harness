"""
Step 8 smoke test — v1 and v2, Haiku, general mode.
Verifies PromptRunner returns structured results for both versions
and that prompt version is correctly stamped on each result object.

Run: python smoke_test_prompts.py
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from dotenv import load_dotenv
from runners.prompt_runner import PromptRunner

load_dotenv()

DATASET_PATH = Path(__file__).parent / "datasets" / "code_eval_dataset.json"

# Limit to 2 non-RAG cases per version to keep cost low
SMOKE_CASE_LIMIT = 2


def main() -> None:
    with open(DATASET_PATH) as f:
        dataset = json.load(f)

    non_rag = [c for c in dataset if c.get("rag_context") is None]
    smoke_dataset = non_rag[:SMOKE_CASE_LIMIT]

    runner = PromptRunner(model="haiku", prompt_versions=["v1", "v2"])
    results_by_version = runner.run(smoke_dataset, mode="general")

    print("\n" + "=" * 60)
    print("SMOKE TEST RESULTS — v1 vs v2 side by side")
    print("=" * 60)

    for version, results in results_by_version.items():
        print(f"\n--- {version} ---")
        for r in results:
            print(f"  [{r['case_id']}] model={r['model']} | prompt={r['prompt_version']} | pass_rate={r['pass_rate']:.2f}")
            for metric, score in r["metric_scores"].items():
                print(f"    {metric}: {score:.3f}")

    # Verify prompt_version field is stamped correctly
    for version, results in results_by_version.items():
        for r in results:
            assert r["prompt_version"] == version, (
                f"Mismatch: expected prompt_version={version}, got {r['prompt_version']}"
            )

    print("\nPrompt version stamping verified. Smoke test complete.\n")


if __name__ == "__main__":
    main()
