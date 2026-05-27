"""
Step 10 — CLI entry point.

Calls ModelRunner and PromptRunner directly — not via the API.
Both cli.py and api/server.py produce identical W&B log entries for identical parameters.

Usage:
  python cli.py --mode general --model haiku --prompt-version v1
  python cli.py --mode rag --model sonnet --prompt-version v2
  python cli.py --mode general --model haiku sonnet --prompt-version v1
  python cli.py --mode general --model haiku --compare-prompts
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from dotenv import load_dotenv
from runners.model_runner import MODEL_MAP, ModelRunner
from runners.prompt_runner import PromptRunner
from runners.retriever import get_retriever
from tracking.wandb_logger import WandbLogger

load_dotenv()

DATASET_PATH = Path(__file__).parent / "datasets" / "code_eval_dataset.json"


def print_summary(all_results: list[dict]) -> None:
    col_widths = {"id": 18, "task_type": 22, "model": 10, "prompt": 8, "metric": 26, "score": 7}
    header = (
        f"{'Case ID':<{col_widths['id']}}"
        f"{'Task Type':<{col_widths['task_type']}}"
        f"{'Model':<{col_widths['model']}}"
        f"{'Prompt':<{col_widths['prompt']}}"
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
            passed = r["metric_pass"][metric_name]
            print(
                f"{r['case_id']:<{col_widths['id']}}"
                f"{r['task_type']:<{col_widths['task_type']}}"
                f"{r['model']:<{col_widths['model']}}"
                f"{r['prompt_version']:<{col_widths['prompt']}}"
                f"{metric_name:<{col_widths['metric']}}"
                f"{score:<{col_widths['score']}.3f}"
                f"{'YES' if passed else 'NO'}"
            )
    print(divider)

    flat_pass = [ok for r in all_results for ok in r["metric_pass"].values()]
    passed_count = sum(flat_pass)
    total = len(flat_pass)
    print(f"\nPassed: {passed_count}/{total} metric checks ({100 * passed_count // total}%)\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="LLM Eval Harness CLI")
    parser.add_argument(
        "--mode",
        choices=["general", "rag"],
        required=True,
        help="Eval mode: 'general' (no retrieval) or 'rag' (retrieval-augmented)",
    )
    parser.add_argument(
        "--model",
        nargs="+",
        choices=list(MODEL_MAP.keys()),
        default=["haiku"],
        help="Model(s) to evaluate. Default: haiku",
    )
    parser.add_argument(
        "--prompt-version",
        nargs="+",
        choices=["v1", "v2", "v3"],
        default=["v1"],
        dest="prompt_versions",
        help="Prompt version(s) to run. Default: v1",
    )
    parser.add_argument(
        "--compare-prompts",
        action="store_true",
        help="Run all three prompt versions and log diffs to W&B",
    )
    args = parser.parse_args()

    with open(DATASET_PATH) as f:
        dataset = json.load(f)

    retriever = get_retriever() if args.mode == "rag" else None
    prompt_versions = ["v1", "v2", "v3"] if args.compare_prompts else args.prompt_versions

    run_label = f"cli-{'-'.join(args.model)}-{'_'.join(prompt_versions)}-{args.mode}"
    logger = WandbLogger(run_name=run_label)
    all_results: list[dict] = []

    try:
        for model_name in args.model:
            if len(prompt_versions) == 1:
                runner = ModelRunner(model=model_name, prompt_version=prompt_versions[0])
                if args.mode == "general":
                    results = runner.run_general(dataset)
                else:
                    results = runner.run_rag(dataset, retriever)
                all_results.extend(results)
            else:
                prompt_runner = PromptRunner(model=model_name, prompt_versions=prompt_versions)
                results_by_version = prompt_runner.run(dataset, mode=args.mode)
                for version_results in results_by_version.values():
                    all_results.extend(version_results)

        logger.log_run(all_results)
    finally:
        logger.finish()

    print_summary(all_results)


if __name__ == "__main__":
    main()
