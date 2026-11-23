import pytest
import pandas as pd
from src.data.collector import GHDataCollector
from src.data.preprocessor import GHDataPreprocessor
from src.models.baseline_popularity import PopularityRecommender
from src.models.baseline_knn import ItemBasedKNNRecommender
from src.models.two_tower import TwoTowerRecommender
from src.models.issue_ranker import IssueSemanticRanker
from src.evaluation.metrics import evaluate_models

def test_full_recsys_pipeline():
    # 1. Dataset Generation
    mock_data = GHDataCollector.generate_mock_dataset(
        num_users=80,
        num_repos=30,
        num_events=1600,
        seed=42
    )
    events = mock_data["events"]
    repos = mock_data["repos"]
    open_issues = mock_data["open_issues"]
    all_repo_names = [r["repo"] for r in repos]

    # 2. Preprocessing & Split
    preprocessor = GHDataPreprocessor(min_user_interactions=2, min_repo_contributors=2)
    df_scores = preprocessor.compute_interaction_scores(events)
    df_filtered = preprocessor.filter_sparse_matrix(df_scores)
    df_train, df_test = preprocessor.create_train_test_split(df_filtered, test_ratio=0.2)
    df_train_negs = preprocessor.generate_negative_samples(df_train, all_repo_names, num_negatives=3)

    # 3. Model Training
    pop_model = PopularityRecommender().fit(df_train, repos)
    knn_model = ItemBasedKNNRecommender().fit(df_train, repos)
    two_tower_model = TwoTowerRecommender(embedding_dim=32, epochs=5).fit(df_train_negs, repos)

    models = {
        "Popularity Baseline": pop_model,
        "Item-based kNN": knn_model,
        "Two-Tower Neural RS": two_tower_model
    }

    # 4. Evaluation
    df_eval = evaluate_models(models, df_train, df_test, all_repo_names, k_values=[5, 10])
    assert not df_eval.empty
    assert len(df_eval) == 3
    print("\n" + df_eval.to_string(index=False))

    # 5. Stage 2 Issue Ranking
    test_user = df_train["user"].iloc[0]
    user_repos = df_train[df_train["user"] == test_user]["repo"].tolist()
    user_repos_meta = [r for r in repos if r["repo"] in user_repos]

    ranked_issues = IssueSemanticRanker().rank_issues_for_user(
        user_past_repos_metadata=user_repos_meta,
        candidate_issues=open_issues,
        top_k=3
    )
    assert len(ranked_issues) == 3
    assert "explanation" in ranked_issues[0]

if __name__ == "__main__":
    test_full_recsys_pipeline()
