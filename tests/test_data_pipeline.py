import os
import pytest
import pandas as pd
from src.data.collector import GHDataCollector
from src.data.preprocessor import GHDataPreprocessor

def test_mock_generation_and_preprocessing():
    # 1. Generate mock data
    mock_data = GHDataCollector.generate_mock_dataset(
        num_users=60,
        num_repos=25,
        num_events=1200,
        seed=42
    )
    events = mock_data["events"]
    repos = mock_data["repos"]
    open_issues = mock_data["open_issues"]

    assert len(events) == 1200
    assert len(repos) == 25
    assert len(open_issues) > 0

    # 2. Preprocess interaction scores
    preprocessor = GHDataPreprocessor(min_user_interactions=3, min_repo_contributors=2)
    df_scores = preprocessor.compute_interaction_scores(events)
    assert not df_scores.empty
    assert "score" in df_scores.columns
    assert "raw_score" in df_scores.columns

    # 3. Filter matrix
    df_filtered = preprocessor.filter_sparse_matrix(df_scores)
    assert len(df_filtered) <= len(df_scores)
    assert not df_filtered.empty

    # 4. Train-test split
    df_train, df_test = preprocessor.create_train_test_split(df_filtered, test_ratio=0.2)
    assert len(df_train) + len(df_test) == len(df_filtered)

    # 5. Negative sampling
    all_repo_names = [r["repo"] for r in repos]
    df_train_neg = preprocessor.generate_negative_samples(df_train, all_repo_names, num_negatives=2)
    assert len(df_train_neg) > len(df_train)
    assert set(df_train_neg["label"].unique()) == {0.0, 1.0}

if __name__ == "__main__":
    test_mock_generation_and_preprocessing()
    print("All data pipeline tests passed successfully!")
