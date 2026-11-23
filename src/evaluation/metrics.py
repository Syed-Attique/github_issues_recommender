"""
Recommender System Evaluation Suite
Implements:
1. Precision@K
2. Recall@K
3. NDCG@K
4. Catalog Coverage
5. Model Benchmarking Runner (Popularity vs kNN vs Two-Tower)
"""

import math
import numpy as np
import pandas as pd
from typing import List, Dict, Set, Any

def precision_at_k(recommended: List[str], ground_truth: Set[str], k: int) -> float:
    """Computes Precision@K."""
    if k <= 0:
        return 0.0
    rec_k = recommended[:k]
    hits = len(set(rec_k) & ground_truth)
    return hits / k

def recall_at_k(recommended: List[str], ground_truth: Set[str], k: int) -> float:
    """Computes Recall@K."""
    if not ground_truth:
        return 0.0
    rec_k = recommended[:k]
    hits = len(set(rec_k) & ground_truth)
    return hits / len(ground_truth)

def ndcg_at_k(recommended: List[str], ground_truth: Set[str], k: int) -> float:
    """Computes NDCG@K for binary relevance."""
    if not ground_truth or k <= 0:
        return 0.0
    rec_k = recommended[:k]

    dcg = 0.0
    for idx, item in enumerate(rec_k):
        if item in ground_truth:
            dcg += 1.0 / math.log2(idx + 2)

    # Ideal DCG
    ideal_hits = min(len(ground_truth), k)
    idcg = sum(1.0 / math.log2(i + 2) for i in range(ideal_hits))
    return dcg / idcg if idcg > 0 else 0.0

def catalog_coverage(all_recommendations: List[List[str]], total_catalog_items: Set[str]) -> float:
    """Computes percentage of unique catalog items recommended at least once."""
    if not total_catalog_items:
        return 0.0
    recommended_set = set()
    for rec_list in all_recommendations:
        recommended_set.update(rec_list)
    return len(recommended_set & total_catalog_items) / len(total_catalog_items)

def evaluate_models(
    models: Dict[str, Any],
    df_train: pd.DataFrame,
    df_test: pd.DataFrame,
    all_catalog_repos: List[str],
    k_values: List[int] = [5, 10]
) -> pd.DataFrame:
    """
    Evaluates multiple recommender models on held-out test data.
    """
    test_user_ground_truth = df_test.groupby("user")["repo"].apply(set).to_dict()
    user_train_history = df_train.groupby("user")["repo"].apply(list).to_dict()
    all_catalog_set = set(all_catalog_repos)

    results = []

    for model_name, model in models.items():
        row = {"Model": model_name}

        for k in k_values:
            precisions = []
            recalls = []
            ndcgs = []
            all_recs_for_k = []

            for user, gt_repos in test_user_ground_truth.items():
                history = user_train_history.get(user, [])
                
                # Model recommendation call
                if hasattr(model, "recommend"):
                    try:
                        # Support models that accept user_history
                        if "user_history" in model.recommend.__code__.co_varnames:
                            recs = model.recommend(user=user, user_history=history, k=k)
                        else:
                            recs = model.recommend(user=user, k=k)
                        rec_repo_names = [r["repo"] for r in recs]
                    except Exception:
                        rec_repo_names = []
                else:
                    rec_repo_names = []

                precisions.append(precision_at_k(rec_repo_names, gt_repos, k))
                recalls.append(recall_at_k(rec_repo_names, gt_repos, k))
                ndcgs.append(ndcg_at_k(rec_repo_names, gt_repos, k))
                all_recs_for_k.append(rec_repo_names)

            row[f"Precision@{k}"] = round(float(np.mean(precisions)), 4)
            row[f"Recall@{k}"] = round(float(np.mean(recalls)), 4)
            row[f"NDCG@{k}"] = round(float(np.mean(ndcgs)), 4)
            row[f"Coverage@{k}"] = round(catalog_coverage(all_recs_for_k, all_catalog_set), 4)

        results.append(row)

    return pd.DataFrame(results)
