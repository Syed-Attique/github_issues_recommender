"""
Popularity-based Baseline Recommender
Recommends the most contributed-to repositories.
Every personalized model must beat this baseline.
"""

import pandas as pd
from typing import List, Dict, Any, Optional

class PopularityRecommender:
    def __init__(self):
        self.popular_repos: List[Dict[str, Any]] = []
        self.user_history: Dict[str, set] = {}

    def fit(self, df_train: pd.DataFrame, repo_metadata: Optional[List[Dict[str, Any]]] = None) -> "PopularityRecommender":
        """
        Calculates repository popularity based on total weighted interaction score
        and unique contributor counts.
        """
        self.user_history = df_train.groupby("user")["repo"].apply(set).to_dict()

        # Aggregate popularity
        repo_stats = df_train.groupby("repo").agg(
            total_score=("score", "sum"),
            unique_contributors=("user", "nunique")
        ).reset_index()

        repo_meta_dict = {}
        if repo_metadata:
            repo_meta_dict = {r["repo"]: r for r in repo_metadata}

        # Sort descending by total score then unique contributors
        sorted_repos = repo_stats.sort_values(
            by=["total_score", "unique_contributors"],
            ascending=False
        )

        self.popular_repos = []
        for _, row in sorted_repos.iterrows():
            repo_name = row["repo"]
            meta = repo_meta_dict.get(repo_name, {})
            self.popular_repos.append({
                "repo": repo_name,
                "score": float(row["total_score"]),
                "contributors": int(row["unique_contributors"]),
                "language": meta.get("language", "Unknown"),
                "topics": meta.get("topics", [])
            })
        return self

    def recommend(
        self,
        user: str,
        k: int = 10,
        filter_language: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Recommends top-k popular repositories excluding those the user has already contributed to.
        """
        seen_repos = self.user_history.get(user, set())
        recommendations = []

        for item in self.popular_repos:
            if item["repo"] in seen_repos:
                continue
            if filter_language and item["language"].lower() != filter_language.lower():
                continue

            recommendations.append({
                "repo": item["repo"],
                "score": item["score"],
                "language": item["language"],
                "topics": item["topics"],
                "explanation": f"Popular baseline: {item['contributors']} distinct contributors in dataset."
            })
            if len(recommendations) >= k:
                break

        return recommendations
