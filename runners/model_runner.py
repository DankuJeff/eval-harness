"""
Step 7 — Multi-model runner.

ModelRunner accepts a list of test cases and a mode ('general' or 'rag'),
runs them against a specified model, and returns structured result objects.

Both evals/general_llm_tests.py and evals/rag_tests.py delegate to this
class instead of calling the Anthropic API directly.
"""

import json
from datetime import datetime, timezone
from pathlib import Path

from anthropic import Anthropic
from deepeval import evaluate
from deepeval.metrics import (
    GEval,
    AnswerRelevancyMetric,
    FaithfulnessMetric,
    HallucinationMetric,
    ContextualPrecisionMetric,
    ContextualRecallMetric,
)
from deepeval.test_case import LLMTestCase, SingleTurnParams

from metrics.custom_metrics import PASS_THRESHOLD

# Short name → full model ID. CLI and API use short names; runner resolves here.
MODEL_MAP: dict[str, str] = {
    "haiku": "claude-haiku-4-5-20251001",
    "sonnet": "claude-sonnet-4-6",
    "opus": "claude-opus-4-7",
}

DATASET_PATH = Path(__file__).parent.parent / "datasets" / "code_eval_dataset.json"

SECURITY_CORRECTNESS_THRESHOLD = 0.8
HALLUCINATION_THRESHOLD = 0.3

GENERAL_PROMPTS: dict[str, str] = {
    "bug_identification": (
        "You are a code reviewer. Analyze the following {language} code and identify any bugs.\n"
        "Describe the bug type, its location in the code, and the root cause.\n\n"
        "Code:\n{code_snippet}"
    ),
    "code_explanation": (
        "You are a code expert. Explain exactly what the following {language} code does.\n"
        "Be precise and faithful to what the code actually does — not what it should do.\n\n"
        "Code:\n{code_snippet}"
    ),
    "code_review": (
        "You are a senior software engineer. Review the following {language} code.\n"
        "Identify issues, suggest specific improvements, and provide clear actionable feedback.\n\n"
        "Code:\n{code_snippet}"
    ),
    "security_flagging": (
        "You are a security expert. Analyze the following {language} code for security vulnerabilities.\n"
        "Focus on the primary vulnerability only. Identify its vulnerability class, attack vector, and recommended fix.\n\n"
        "Code:\n{code_snippet}"
    ),
}

RAG_PROMPTS: dict[str, str] = {
    "bug_identification": (
        "You are a code reviewer. Use the reference documentation below to help identify bugs.\n\n"
        "Reference:\n{context}\n\n"
        "Analyze the following {language} code and identify any bugs.\n"
        "Describe the bug type, its location, and the root cause.\n\n"
        "Code:\n{code_snippet}"
    ),
    "code_explanation": (
        "You are a code expert. Use the reference documentation below to inform your explanation.\n\n"
        "Reference:\n{context}\n\n"
        "Explain exactly what the following {language} code does.\n"
        "Be precise and faithful to what the code actually does.\n\n"
        "Code:\n{code_snippet}"
    ),
    "code_review": (
        "You are a senior software engineer. Use the reference documentation below in your review.\n\n"
        "Reference:\n{context}\n\n"
        "Review the following {language} code. Identify issues and suggest specific improvements.\n\n"
        "Code:\n{code_snippet}"
    ),
    "security_flagging": (
        "You are a security expert. Use the reference documentation below to inform your analysis.\n\n"
        "Reference:\n{context}\n\n"
        "Analyze the following {language} code for security vulnerabilities.\n"
        "Focus on the primary vulnerability only.\n\n"
        "Code:\n{code_snippet}"
    ),
}


def _get_general_metrics(task_type: str) -> list:
    if task_type == "bug_identification":
        return [
            GEval(
                name="Correctness",
                criteria=(
                    "Determine whether the response correctly identifies the bug type, "
                    "its location in the code, and the root cause, as described in the expected output."
                ),
                evaluation_params=[
                    SingleTurnParams.INPUT,
                    SingleTurnParams.ACTUAL_OUTPUT,
                    SingleTurnParams.EXPECTED_OUTPUT,
                ],
                threshold=PASS_THRESHOLD,
            ),
            AnswerRelevancyMetric(threshold=PASS_THRESHOLD),
        ]
    if task_type == "code_explanation":
        return [
            FaithfulnessMetric(threshold=PASS_THRESHOLD),
            AnswerRelevancyMetric(threshold=PASS_THRESHOLD),
        ]
    if task_type == "code_review":
        return [
            GEval(
                name="ReviewQuality",
                criteria=(
                    "Evaluate the code review across three criteria: "
                    "(1) whether identified issues are accurate and complete, "
                    "(2) whether suggested improvements are specific and actionable, "
                    "(3) whether the feedback is clearly written and easy to act on."
                ),
                evaluation_params=[
                    SingleTurnParams.INPUT,
                    SingleTurnParams.ACTUAL_OUTPUT,
                ],
                threshold=PASS_THRESHOLD,
            ),
            GEval(
                name="Coherence",
                criteria=(
                    "Determine whether the code review is logically structured, "
                    "internally consistent, and free of contradictions."
                ),
                evaluation_params=[
                    SingleTurnParams.INPUT,
                    SingleTurnParams.ACTUAL_OUTPUT,
                ],
                threshold=PASS_THRESHOLD,
            ),
        ]
    if task_type == "security_flagging":
        return [
            GEval(
                name="Correctness",
                criteria=(
                    "Determine whether the response correctly identifies the vulnerability class, "
                    "the attack vector, and the recommended fix, as described in the expected output."
                ),
                evaluation_params=[
                    SingleTurnParams.INPUT,
                    SingleTurnParams.ACTUAL_OUTPUT,
                    SingleTurnParams.EXPECTED_OUTPUT,
                ],
                threshold=SECURITY_CORRECTNESS_THRESHOLD,
            ),
            HallucinationMetric(threshold=HALLUCINATION_THRESHOLD),
        ]
    raise ValueError(f"Unknown task_type: {task_type}")


def _get_rag_metrics() -> list:
    return [
        ContextualPrecisionMetric(threshold=PASS_THRESHOLD),
        ContextualRecallMetric(threshold=PASS_THRESHOLD),
        FaithfulnessMetric(threshold=PASS_THRESHOLD),
        AnswerRelevancyMetric(threshold=PASS_THRESHOLD),
    ]


class ModelRunner:
    def __init__(self, model: str, prompt_version: str = "v1"):
        if model not in MODEL_MAP:
            raise ValueError(f"Unknown model '{model}'. Valid options: {list(MODEL_MAP.keys())}")
        self.model_short = model
        self.model_id = MODEL_MAP[model]
        self.prompt_version = prompt_version
        self._client = Anthropic()

    def _call_model(self, prompt: str) -> str:
        response = self._client.messages.create(
            model=self.model_id,
            max_tokens=1024,
            messages=[{"role": "user", "content": prompt}],
        )
        return response.content[0].text

    def run_general(self, dataset: list[dict]) -> list[dict]:
        """Run general (non-RAG) eval. Groups by task type for metric consistency."""
        non_rag = [c for c in dataset if c.get("rag_context") is None]
        by_type: dict[str, list] = {}
        for case in non_rag:
            by_type.setdefault(case["task_type"], []).append(case)

        all_results: list[dict] = []
        for task_type, cases in by_type.items():
            metrics = _get_general_metrics(task_type)
            test_cases: list[LLMTestCase] = []
            case_meta: list[dict] = []

            for case in cases:
                prompt = GENERAL_PROMPTS[task_type].format(
                    language=case["input"]["language"],
                    code_snippet=case["input"]["code_snippet"],
                )
                actual_output = self._call_model(prompt)
                test_cases.append(
                    LLMTestCase(
                        input=prompt,
                        actual_output=actual_output,
                        expected_output=case["ground_truth"]["expected_output"],
                        retrieval_context=[case["input"]["code_snippet"]],
                        context=[case["input"]["code_snippet"]],
                    )
                )
                case_meta.append({"case_id": case["id"], "task_type": task_type})

            eval_results = evaluate(test_cases=test_cases, metrics=metrics)
            timestamp = datetime.now(timezone.utc).isoformat()

            for meta, test_result in zip(case_meta, eval_results.test_results):
                metric_scores = {
                    md.name: md.score for md in test_result.metrics_data
                }
                metric_pass = {
                    md.name: md.success for md in test_result.metrics_data
                }
                passed_count = sum(metric_pass.values())
                all_results.append(
                    {
                        "case_id": meta["case_id"],
                        "task_type": meta["task_type"],
                        "model": self.model_short,
                        "model_id": self.model_id,
                        "prompt_version": self.prompt_version,
                        "mode": "general",
                        "metric_scores": metric_scores,
                        "metric_pass": metric_pass,
                        "pass_rate": passed_count / len(test_result.metrics_data),
                        "timestamp": timestamp,
                    }
                )
        return all_results

    def run_rag(self, dataset: list[dict], retriever) -> list[dict]:
        """Run RAG eval. All 8 RAG cases share the same 4 metrics."""
        rag_cases = [c for c in dataset if c.get("rag_context") is not None]
        metrics = _get_rag_metrics()
        test_cases: list[LLMTestCase] = []
        case_meta: list[dict] = []

        for case in rag_cases:
            task_type = case["task_type"]
            retrieved = retriever.retrieve(case["input"]["code_snippet"], top_k=3)
            retrieved_texts = [r["context"] for r in retrieved]
            context_block = "\n\n".join(retrieved_texts)

            prompt = RAG_PROMPTS[task_type].format(
                language=case["input"]["language"],
                code_snippet=case["input"]["code_snippet"],
                context=context_block,
            )
            actual_output = self._call_model(prompt)
            test_cases.append(
                LLMTestCase(
                    input=prompt,
                    actual_output=actual_output,
                    expected_output=case["ground_truth"]["expected_output"],
                    retrieval_context=retrieved_texts,
                )
            )
            case_meta.append({"case_id": case["id"], "task_type": task_type})

        eval_results = evaluate(test_cases=test_cases, metrics=metrics)
        timestamp = datetime.now(timezone.utc).isoformat()

        all_results: list[dict] = []
        for meta, test_result in zip(case_meta, eval_results.test_results):
            metric_scores = {
                md.name: md.score for md in test_result.metrics_data
            }
            metric_pass = {
                md.name: md.success for md in test_result.metrics_data
            }
            passed_count = sum(metric_pass.values())
            all_results.append(
                {
                    "case_id": meta["case_id"],
                    "task_type": meta["task_type"],
                    "model": self.model_short,
                    "model_id": self.model_id,
                    "prompt_version": self.prompt_version,
                    "mode": "rag",
                    "metric_scores": metric_scores,
                    "metric_pass": metric_pass,
                    "pass_rate": passed_count / len(test_result.metrics_data),
                    "timestamp": timestamp,
                }
            )
        return all_results
