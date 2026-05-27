"""
Step 4 / Step 7 — General LLM eval tests.

Loads all non-RAG test cases from the dataset and runs them through ModelRunner.
Prints a summary table to stdout. No W&B logging until Step 9.

Run directly: python evals/general_llm_tests.py
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from dotenv import load_dotenv
from runners.model_runner import ModelRunner

load_dotenv()

DATASET_PATH = Path(__file__).parent.parent / "datasets" / "code_eval_dataset.json"


def print_summary(all_results: list[dict]) -> None:
    col_widths = {"id": 18, "task_type": 22, "metric": 20, "score": 7, "pass": 5}
    header = (
        f"{'Case ID':<{col_widths['id']}}"
        f"{'Task Type':<{col_widths['task_type']}}"
        f"{'Metric':<{col_widths['metric']}}"
        f"{'Score':<{col_widths['score']}}"
        f"{'Pass'}"
    )
    divider = "-" * len(header)
    print(f"\n{divider}")
    print(header)
    print(divider)
    for r in all_results:
        for metric_name, score in r["metric_scores"].items():
            passed = score >= 0.7
            print(
                f"{r['case_id']:<{col_widths['id']}}"
                f"{r['task_type']:<{col_widths['task_type']}}"
                f"{metric_name:<{col_widths['metric']}}"
                f"{score:<{col_widths['score']}.3f}"
                f"{'YES' if passed else 'NO'}"
            )
    print(divider)

    flat = [
        score
        for r in all_results
        for score in r["metric_scores"].values()
    ]
    passed = sum(1 for score in flat if score >= 0.7)
    total = len(flat)
    print(f"\nModel: {all_results[0]['model_id']}  |  Prompt: {all_results[0]['prompt_version']}")
    print(f"Passed: {passed}/{total} metric checks ({100 * passed // total}%)\n")


def main() -> None:
    with open(DATASET_PATH) as f:
        dataset = json.load(f)

    non_rag = [c for c in dataset if c.get("rag_context") is None]
    print(f"Loaded {len(non_rag)} non-RAG test cases.")

    runner = ModelRunner(model="haiku", prompt_version="v1")
    results = runner.run_general(dataset)
    print_summary(results)


if __name__ == "__main__":
    main()
