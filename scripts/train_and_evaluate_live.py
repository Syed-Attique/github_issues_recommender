"""
Train and Evaluate Models on Live GitHub Data
Loads data/raw/live_github_data.json, preprocesses user-repo interactions,
trains Popularity, Item-based kNN, and Two-Tower Neural RS,
and outputs comparative evaluation metrics.
"""

import os
import json
import pandas as pd
from src.data.preprocessor import GHDataPreprocessor
from src.models.baseline_popularity import PopularityRecommender
from src.models.baseline_knn import ItemBasedKNNRecommender
from src.models.two_tower import TwoTowerRecommender
from src.evaluation.metrics import evaluate_models

def run_live_training_and_evaluation(raw_data_path: str = "data/raw/live_github_data.json"):
    if not os.path.exists(raw_data_path):
        print(f"File {raw_data_path} not found.")
        return

    with open(raw_data_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    events = data.get("events", [])
    repos = data.get("repos", [])
    open_issues = data.get("open_issues", [])
    all_repo_names = [r["repo"] for r in repos]

    print(f"Loaded {len(repos)} repositories and {len(events)} interactions from live dataset.")

    # 1. Preprocessing
    preprocessor = GHDataPreprocessor(min_user_interactions=1, min_repo_contributors=1)
    df_scores = preprocessor.compute_interaction_scores(events)
    df_filtered = preprocessor.filter_sparse_matrix(df_scores)
    df_train, df_test = preprocessor.create_train_test_split(df_filtered, test_ratio=0.25)
    df_train_negs = preprocessor.generate_negative_samples(df_train, all_repo_names, num_negatives=3)

    print(f"Training interactions: {len(df_train)} (with negatives: {len(df_train_negs)}) | Test interactions: {len(df_test)}")

    # 2. Train Models
    print("Training Popularity Baseline...")
    pop_model = PopularityRecommender().fit(df_train, repos)

    print("Training Item-based kNN...")
    knn_model = ItemBasedKNNRecommender().fit(df_train, repos)

    print("Training Two-Tower Neural RS...")
    two_tower_model = TwoTowerRecommender(embedding_dim=32, epochs=10).fit(df_train_negs, repos)

    # 3. Benchmark
    models = {
        "Popularity Baseline": pop_model,
        "Item-based kNN": knn_model,
        "Two-Tower Neural RS": two_tower_model
    }

    print("\n--- Evaluating Models on Live Data Split ---")
    df_eval = evaluate_models(models, df_train, df_test, all_repo_names, k_values=[3, 5, 10])
    print(df_eval.to_string(index=False))

    # Save processed split and eval results
    os.makedirs("data/processed", exist_ok=True)
    df_eval.to_csv("data/processed/live_evaluation_results.csv", index=False)
    print("\nSaved evaluation results to data/processed/live_evaluation_results.csv")

    return df_eval

if __name__ == "__main__":
    run_live_training_and_evaluation()
