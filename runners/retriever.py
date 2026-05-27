import json
import os
import numpy as np
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

_client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])

DATASET_PATH = os.path.join(os.path.dirname(__file__), "..", "datasets", "code_eval_dataset.json")

_EMBED_MODEL = "text-embedding-3-small"


def _embed(texts: list[str]) -> np.ndarray:
    response = _client.embeddings.create(model=_EMBED_MODEL, input=texts)
    return np.array([item.embedding for item in response.data], dtype=np.float32)


def _cosine_similarity(query_vec: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    query_norm = query_vec / (np.linalg.norm(query_vec) + 1e-10)
    row_norms = np.linalg.norm(matrix, axis=1, keepdims=True) + 1e-10
    normalized = matrix / row_norms
    return normalized @ query_norm


class Retriever:
    def __init__(self):
        with open(DATASET_PATH, "r", encoding="utf-8") as f:
            dataset = json.load(f)

        self._chunks: list[dict] = [
            {"id": case["id"], "task_type": case["task_type"], "context": case["rag_context"]}
            for case in dataset
            if case.get("rag_context") is not None
        ]

        contexts = [chunk["context"] for chunk in self._chunks]
        self._matrix = _embed(contexts)

    def retrieve(self, query: str, top_k: int = 3) -> list[dict]:
        query_vec = _embed([query])[0]
        scores = _cosine_similarity(query_vec, self._matrix)
        top_indices = np.argsort(scores)[::-1][:top_k]
        results = []
        for idx in top_indices:
            chunk = self._chunks[idx].copy()
            chunk["score"] = float(scores[idx])
            results.append(chunk)
        return results


_retriever: Retriever | None = None


def get_retriever() -> Retriever:
    global _retriever
    if _retriever is None:
        _retriever = Retriever()
    return _retriever
