# Metrics Reference

This document defines the evaluation metrics used by the harness, how DeepEval scores them, and what threshold determines a pass.

---

## Pass Threshold

**0.7** across all metrics. A test case passes a metric if its score is ≥ 0.7. A test case is considered fully passing only if all applicable metrics pass.

---

## What LLM-as-Judge Means

Several metrics in this harness use an LLM (DeepEval's default OpenAI judge, resolved from `OPENAI_API_KEY`) to evaluate the quality of another LLM's output. This is called LLM-as-judge. The judge model is not pinned in code — DeepEval selects it and tracks OpenAI releases, so it can change over time (this project started on `gpt-4o` and DeepEval auto-upgraded to `gpt-5.4` mid-run).

Instead of checking for an exact string match against ground truth — which fails for open-ended responses — the judge model reads the question, the response, and the reference answer, then scores the response on defined criteria. This enables evaluation of subjective qualities like correctness of reasoning, faithfulness to source material, and coherence.

All scores are floats between 0.0 and 1.0. DeepEval handles the judge model calls internally; the harness only needs a valid `OPENAI_API_KEY`.

---

## Metrics by Task Type

### bug_identification

| Metric | Type | What It Measures |
|--------|------|-----------------|
| Correctness | GEval (LLM-as-judge) | Whether the model correctly identified the bug type, location, and root cause against the ground truth expected output |
| Answer Relevancy | LLM-as-judge | Whether the response stays focused on the bug in question and does not include off-topic content |

**Why these:** Bug identification has a correct answer — either you found the bug or you didn't. GEval with a correctness criterion captures partial credit (e.g., identified the right location but wrong cause). Answer relevancy catches verbose responses that bury the answer in irrelevant content.

---

### code_explanation

| Metric | Type | What It Measures |
|--------|------|-----------------|
| Faithfulness | LLM-as-judge | Whether every claim in the explanation is grounded in what the code actually does — no hallucinated behavior |
| Answer Relevancy | LLM-as-judge | Whether the explanation addresses what the code does, not tangential topics |

**Why these:** Code explanation is the highest hallucination risk task — models frequently describe what code *should* do rather than what it *actually* does. Faithfulness directly penalizes this. Answer relevancy ensures the explanation doesn't drift into tangential API documentation or unrelated concepts.

---

### code_review

| Metric | Type | What It Measures |
|--------|------|-----------------|
| G-Eval (custom criteria) | LLM-as-judge | Scores the review across three axes: (1) correctness of identified issues, (2) quality and specificity of suggested improvements, (3) clarity and actionability of the feedback |
| Coherence | GEval (LLM-as-judge) | Whether the review is logically structured, internally consistent, and does not contradict itself |

**Why these:** Code review is the most subjective task type — there is no single correct answer. GEval with explicit criteria is the right tool because it lets us define what "good" looks like rather than comparing to a reference string. Coherence catches reviews that are technically accurate but poorly structured or self-contradictory.

---

### security_flagging

| Metric | Type | What It Measures |
|--------|------|-----------------|
| Correctness | GEval (LLM-as-judge) | Whether the model correctly identified the vulnerability class (e.g., SQL injection, XSS), the attack vector, and the recommended fix |
| Hallucination | LLM-as-judge | Whether the model invents vulnerabilities that do not exist in the code, or describes attack vectors that are not applicable |

**Why these:** Security flagging has high stakes for both false negatives (missing a real vulnerability) and false positives (inventing one). Correctness captures whether the real issue was found. Hallucination specifically penalizes invented vulnerabilities, which is the primary failure mode for security-focused LLM responses.

---

### RAG Mode (all task types)

When a test case is run in RAG mode, the following four metrics are applied regardless of task type. They evaluate both retrieval quality and generation quality.

| Metric | Type | What It Measures |
|--------|------|-----------------|
| Contextual Precision | LLM-as-judge | Whether the retrieved context chunks that were used are actually relevant to the query — penalizes retrieving irrelevant chunks |
| Contextual Recall | LLM-as-judge | Whether all information needed to answer the query was present in the retrieved chunks — penalizes missing relevant content |
| Faithfulness | LLM-as-judge | Whether the generated answer is grounded in the retrieved context and does not introduce outside information |
| Answer Relevancy | LLM-as-judge | Whether the generated answer addresses the actual question asked |

**Why these four:** RAG evaluation has two distinct failure surfaces — retrieval and generation. Contextual precision and recall measure retrieval quality. Faithfulness and answer relevancy measure generation quality given the retrieved context. A system can retrieve perfectly and generate poorly, or retrieve poorly and still generate an acceptable answer from partial context — these four metrics expose both failure modes independently.

---

## DeepEval Scoring Summary

| Metric | Score Range | Scoring Method |
|--------|-------------|----------------|
| GEval (correctness, coherence, custom) | 0.0 – 1.0 | LLM judge evaluates against explicit criteria |
| AnswerRelevancyMetric | 0.0 – 1.0 | LLM judge splits the answer into statements and scores the fraction relevant to the question |
| FaithfulnessMetric | 0.0 – 1.0 | LLM judge checks each claim against source |
| HallucinationMetric | 0.0 – 1.0 | LLM judge detects claims not supported by context |
| ContextualPrecisionMetric | 0.0 – 1.0 | LLM judge scores relevance of retrieved chunks |
| ContextualRecallMetric | 0.0 – 1.0 | LLM judge scores coverage of retrieved chunks |

All metrics use 0.7 as the pass threshold.

---

## Threshold Notes for Step 4 Wiring

- **HallucinationMetric** uses an inverted scale (higher score = more hallucination, pass criterion is `score ≤ threshold`). Create a separate `HALLUCINATION_THRESHOLD = 0.3` constant — do NOT use `PASS_THRESHOLD` for this metric.
- **security_flagging correctness** — bump GEval correctness threshold to 0.8 or 0.85. Higher stakes than other task types; 0.7 is too lenient for missed vulnerabilities.
- **AnswerRelevancyMetric** — monitor scores in Step 4 runs. Answer relevancy tends to score high by default. If the metric never fails anything useful, introduce a separate `ANSWER_RELEVANCY_THRESHOLD` and tighten it.
- **GEval (all uses)** — monitor scores in Step 4 runs. If 0.7 is passing marginal answers that clearly shouldn't pass, introduce a separate `GEVAL_THRESHOLD` and adjust. Adjust after seeing real data, not before.
