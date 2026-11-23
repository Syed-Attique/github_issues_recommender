import pytest
from src.data.collector import GHDataCollector
from src.data.preprocessor import GHDataPreprocessor
from src.models.baseline_popularity import PopularityRecommender
from src.models.baseline_knn import ItemBasedKNNRecommender

@pytest.fixture
def sample_data():
    mock_data = GHDataCollector.generate_mock_dataset(num_users=40, num_repos=20, num_events=800, seed=42)
    preprocessor = GHDataPreprocessor(min_user_interactions=2, min_repo_contributors=2)
    df_scores = preprocessor.compute_interaction_scores(mock_data["events"])
    df_filtered = preprocessor.filter_sparse_matrix(df_scores)
    df_train, df_test = preprocessor.create_train_test_split(df_filtered, test_ratio=0.2)
    return df_train, df_test, mock_data["repos"]

def test_popularity_recommender(sample_data):
    df_train, _, repos = sample_data
    pop_model = PopularityRecommender().fit(df_train, repos)
    recs = pop_model.recommend(user="developer_0", k=5)
    assert len(recs) <= 5
    if len(recs) > 0:
        assert "score" in recs[0]
        assert "explanation" in recs[0]
        assert recs[0]["score"] >= recs[-1]["score"]

def test_knn_recommender(sample_data):
    df_train, _, repos = sample_data
    knn_model = ItemBasedKNNRecommender().fit(df_train, repos)
    user = df_train["user"].iloc[0]
    recs = knn_model.recommend(user=user, k=5)
    if len(recs) > 0:
        assert "repo" in recs[0]
        assert "score" in recs[0]
        assert "explanation" in recs[0]
