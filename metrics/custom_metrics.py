"""
Custom metric definitions for the LLM eval harness.

Metric plan by task type:

| Task Type         | Primary Metric                  | Secondary Metric    |
|-------------------|---------------------------------|---------------------|
| bug_identification| Correctness (LLM-as-judge)      | Answer relevancy    |
| code_explanation  | Faithfulness                    | Answer relevancy    |
| code_review       | G-Eval (custom criteria)        | Coherence           |
| security_flagging | Correctness (LLM-as-judge)      | Hallucination       |
| RAG modes (all)   | Contextual precision            | Contextual recall   |

Pass threshold: 0.7 across all metrics (scores range 0.0 to 1.0).

Implementation happens in Step 4 (general LLM metrics) and Step 6 (RAG metrics).
"""

from deepeval.metrics import (
    GEval,
    AnswerRelevancyMetric,
    FaithfulnessMetric,
    HallucinationMetric,
    ContextualPrecisionMetric,
    ContextualRecallMetric,
)
from deepeval.test_case import LLMTestCase, LLMTestCaseParams

PASS_THRESHOLD = 0.7

# Step 4 wiring notes:
# - HALLUCINATION_THRESHOLD = 0.3 (inverted scale — pass if score <= threshold, NOT >= threshold)
# - security_flagging GEval correctness: use 0.8 or 0.85, not PASS_THRESHOLD
# - AnswerRelevancyMetric: watch Step 4 runs — add ANSWER_RELEVANCY_THRESHOLD if 0.7 too lenient
# - GEval (all): watch Step 4 runs — add GEVAL_THRESHOLD if 0.7 passing marginal answers


class BugIdentificationMetrics:
    """
    Metrics for bug_identification task type.

    Primary:   Correctness via GEval — judges whether the model correctly
               identified the bug type, location, and cause against ground truth.
    Secondary: AnswerRelevancyMetric — checks that the response stays on topic
               and does not include irrelevant explanation.
    """
    pass


class CodeExplanationMetrics:
    """
    Metrics for code_explanation task type.

    Primary:   FaithfulnessMetric — checks that all claims in the explanation
               are grounded in the actual code, with no hallucinated behavior.
    Secondary: AnswerRelevancyMetric — checks that the explanation addresses
               what the code does, not tangential topics.
    """
    pass


class CodeReviewMetrics:
    """
    Metrics for code_review task type.

    Primary:   GEval with custom criteria — judges the response across three
               axes: (1) correctness of identified issues, (2) quality of
               suggested improvements, (3) clarity and actionability.
    Secondary: GEval coherence — checks that the review is logically structured
               and internally consistent.
    """
    pass


class SecurityFlaggingMetrics:
    """
    Metrics for security_flagging task type.

    Primary:   Correctness via GEval — judges whether the model correctly
               identified the vulnerability class, attack vector, and fix
               against ground truth.
    Secondary: HallucinationMetric — checks that the model does not invent
               vulnerabilities or describe non-existent attack vectors.
    """
    pass


class RAGMetrics:
    """
    Metrics applied to all task types when run in RAG mode.

    These replace (not supplement) the task-specific metrics for RAG test cases.

    Primary:   ContextualPrecisionMetric — measures whether the retrieved
               context chunks that were used are actually relevant to the query.
               Penalizes retrieving irrelevant chunks even if the answer is correct.
    Secondary: ContextualRecallMetric — measures whether all the information
               needed to answer the query was present in the retrieved context.
               Penalizes missing relevant chunks.

    Additional metrics applied alongside:
    - FaithfulnessMetric — ensures the generated answer is grounded in the
      retrieved context and does not introduce information from outside it.
    - AnswerRelevancyMetric — ensures the answer addresses the actual question.
    """
    pass
