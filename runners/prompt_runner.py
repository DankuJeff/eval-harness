"""
Step 8 — Prompt versioning and regression runner.

Defines three prompt versions per task type and a PromptRunner class that
runs the full eval suite once per version, returning all results grouped by version.

Prompt strategies:
  v1 — direct instruction
  v2 — chain-of-thought (think step by step)
  v3 — role-primed (expert persona with explicit scope constraints)
"""

import json
from pathlib import Path

from runners.model_runner import ModelRunner, MODEL_MAP

DATASET_PATH = Path(__file__).parent.parent / "datasets" / "code_eval_dataset.json"

# Prompt registry: task_type → version → template string
PROMPT_REGISTRY: dict[str, dict[str, str]] = {
    "bug_identification": {
        "v1": (
            "You are a code reviewer. Analyze the following {language} code and identify any bugs.\n"
            "Describe the bug type, its location in the code, and the root cause.\n\n"
            "Code:\n{code_snippet}"
        ),
        "v2": (
            "You are a code reviewer. Think step by step.\n"
            "First, read the following {language} code carefully from top to bottom.\n"
            "Then identify any bugs — focus only on the bug type, its location, and the root cause.\n\n"
            "Code:\n{code_snippet}"
        ),
        "v3": (
            "You are a senior software engineer specializing in code correctness.\n"
            "Your task is strictly to identify bugs — not to suggest fixes, refactor, or explain what the code does.\n"
            "For the following {language} code, state: (1) the bug type, (2) the exact location, (3) the root cause.\n\n"
            "Code:\n{code_snippet}"
        ),
    },
    "code_explanation": {
        "v1": (
            "You are a code expert. Explain exactly what the following {language} code does.\n"
            "Be precise and faithful to what the code actually does — not what it should do.\n\n"
            "Code:\n{code_snippet}"
        ),
        "v2": (
            "You are a code expert. Think step by step.\n"
            "Read the following {language} code carefully, then explain exactly what it does.\n"
            "Trace the execution path and be faithful to what the code actually does.\n\n"
            "Code:\n{code_snippet}"
        ),
        "v3": (
            "You are a technical writer who explains code to engineers.\n"
            "Your task is strictly to describe what the following {language} code does — not to evaluate it, suggest improvements, or explain what it should do.\n"
            "Be precise and grounded only in the actual behavior of the code.\n\n"
            "Code:\n{code_snippet}"
        ),
    },
    "code_review": {
        "v1": (
            "You are a senior software engineer. Review the following {language} code.\n"
            "Identify issues, suggest specific improvements, and provide clear actionable feedback.\n\n"
            "Code:\n{code_snippet}"
        ),
        "v2": (
            "You are a senior software engineer. Think step by step.\n"
            "First read the following {language} code carefully.\n"
            "Then identify issues across three areas: correctness, design, and readability.\n"
            "For each issue, suggest a specific, actionable improvement.\n\n"
            "Code:\n{code_snippet}"
        ),
        "v3": (
            "You are a staff engineer conducting a formal code review.\n"
            "Review the following {language} code across three dimensions: "
            "(1) correctness — bugs or logic errors, "
            "(2) design — structure, responsibility, and coupling, "
            "(3) readability — naming, clarity, and maintainability.\n"
            "For each issue found, provide one specific, actionable suggestion.\n\n"
            "Code:\n{code_snippet}"
        ),
    },
    "security_flagging": {
        "v1": (
            "You are a security expert. Analyze the following {language} code for security vulnerabilities.\n"
            "Focus on the primary vulnerability only. Identify its vulnerability class, attack vector, and recommended fix.\n\n"
            "Code:\n{code_snippet}"
        ),
        "v2": (
            "You are a security expert. Think step by step.\n"
            "Read the following {language} code carefully, then identify the primary security vulnerability.\n"
            "Report only the most critical issue: its vulnerability class, attack vector, and recommended fix.\n\n"
            "Code:\n{code_snippet}"
        ),
        "v3": (
            "You are an application security engineer conducting a focused vulnerability assessment.\n"
            "Identify the single most critical security vulnerability in the following {language} code.\n"
            "Your response must cover exactly three things: (1) vulnerability class, (2) attack vector, (3) recommended fix.\n"
            "Do not list secondary issues.\n\n"
            "Code:\n{code_snippet}"
        ),
    },
}


class PromptRunner:
    """Runs the full eval suite across one or more prompt versions for a given model."""

    def __init__(self, model: str, prompt_versions: list[str] | None = None):
        if model not in MODEL_MAP:
            raise ValueError(f"Unknown model '{model}'. Valid options: {list(MODEL_MAP.keys())}")
        self.model = model
        self.prompt_versions = prompt_versions or ["v1", "v2", "v3"]

    def run(self, dataset: list[dict], mode: str = "general") -> dict[str, list[dict]]:
        """
        Run eval for each prompt version. Returns results grouped by version.

        Args:
            dataset: Full dataset list (runner filters by mode internally).
            mode: 'general' or 'rag'. RAG mode uses ModelRunner's RAG prompts (not this registry).

        Returns:
            { "v1": [result, ...], "v2": [result, ...], ... }
        """
        if mode not in ("general", "rag"):
            raise ValueError(f"Unknown mode '{mode}'. Must be 'general' or 'rag'.")

        results_by_version: dict[str, list[dict]] = {}

        for version in self.prompt_versions:
            print(f"\nRunning {self.model.upper()} / {version} / {mode}...")
            runner = ModelRunner(model=self.model, prompt_version=version)

            # Patch the prompt templates for this version before running
            if mode == "general":
                _apply_prompt_version(runner, version)
                version_results = runner.run_general(dataset)
            else:
                version_results = runner.run_rag(dataset, _get_retriever())

            results_by_version[version] = version_results

        return results_by_version


def _apply_prompt_version(runner: ModelRunner, version: str) -> None:
    """Patch ModelRunner's GENERAL_PROMPTS with the versioned templates from this registry."""
    from runners import model_runner
    for task_type, versions in PROMPT_REGISTRY.items():
        if version in versions:
            model_runner.GENERAL_PROMPTS[task_type] = versions[version]


def _get_retriever():
    from runners.retriever import get_retriever
    return get_retriever()
