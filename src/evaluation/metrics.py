import json
import logging
from pathlib import Path

import numpy as np
from sklearn.metrics import fbeta_score, precision_score, recall_score

logger = logging.getLogger(__name__)


def compute_f2_macro(y_true: list[list[str]], y_pred: list[list[str]]) -> dict:
    """Compute F2-Macro for legal retrieval/QA.

    y_true: list of list of relevant article IDs per query
    y_pred: list of list of predicted article IDs per query
    """
    all_articles = sorted(set(a for articles in y_true for a in articles))
    article_to_idx = {a: i for i, a in enumerate(all_articles)}
    n = len(y_true)
    n_articles = len(all_articles)

    y_true_bin = np.zeros((n, n_articles), dtype=int)
    y_pred_bin = np.zeros((n, n_articles), dtype=int)

    for i in range(n):
        for a in y_true[i]:
            if a in article_to_idx:
                y_true_bin[i, article_to_idx[a]] = 1
        for a in y_pred[i]:
            if a in article_to_idx:
                y_pred_bin[i, article_to_idx[a]] = 1

    per_query = []
    for i in range(n):
        f2 = fbeta_score(y_true_bin[i], y_pred_bin[i], beta=2, zero_division=0)
        per_query.append(f2)

    macro_f2 = np.mean(per_query)
    micro_precision = precision_score(y_true_bin, y_pred_bin, average="micro", zero_division=0)
    micro_recall = recall_score(y_true_bin, y_pred_bin, average="micro", zero_division=0)
    micro_f2 = fbeta_score(y_true_bin, y_pred_bin, beta=2, average="micro", zero_division=0)

    return {
        "macro_f2": float(macro_f2),
        "micro_f2": float(micro_f2),
        "micro_precision": float(micro_precision),
        "micro_recall": float(micro_recall),
        "num_queries": n,
        "num_articles": n_articles,
        "per_query_f2": [float(f) for f in per_query],
    }


def compute_retrieval_metrics(
    y_true: list[list[str]],
    y_pred: list[list[str]],
    k_values: list[int] = [5, 10, 20, 50],
) -> dict:
    results = {}
    for k in k_values:
        recalls = []
        precisions = []
        for true_articles, pred_articles in zip(y_true, y_pred):
            pred_k = pred_articles[:k]
            if len(true_articles) == 0:
                continue
            hits = len(set(pred_k) & set(true_articles))
            recalls.append(hits / len(true_articles))
            precisions.append(hits / k)

        results[f"recall@{k}"] = float(np.mean(recalls)) if recalls else 0.0
        results[f"precision@{k}"] = float(np.mean(precisions)) if precisions else 0.0

    return results


def compute_mrr(y_true: list[list[str]], y_pred: list[list[str]]) -> float:
    reciprocal_ranks = []
    for true_articles, pred_articles in zip(y_true, y_pred):
        true_set = set(true_articles)
        if not true_set:
            continue
        rr = 0.0
        for rank, article in enumerate(pred_articles, start=1):
            if article in true_set:
                rr = 1.0 / rank
                break
        reciprocal_ranks.append(rr)
    return float(np.mean(reciprocal_ranks)) if reciprocal_ranks else 0.0


def evaluate_answers(
    ground_truth: list[dict],
    predictions: list[dict],
    output_path: str | Path | None = None,
) -> dict:
    y_true_articles = []
    y_pred_articles = []

    for gt, pred in zip(ground_truth, predictions):
        y_true_articles.append(gt.get("relevant_articles", []))
        y_pred_articles.append(pred.get("relevant_articles", []))

    f2_metrics = compute_f2_macro(y_true_articles, y_pred_articles)
    retrieval_metrics = compute_retrieval_metrics(y_true_articles, y_pred_articles)
    mrr = compute_mrr(y_true_articles, y_pred_articles)

    results = {**f2_metrics, **retrieval_metrics, "mrr": mrr}

    if output_path:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        logger.info(f"Evaluation results saved to {output_path}")

    return results
