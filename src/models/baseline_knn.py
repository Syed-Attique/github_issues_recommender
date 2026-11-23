"""
Item-Based kNN Collaborative Filtering Recommender
Computes item-item cosine similarity over user-repository interaction matrix.
Generates explainable recommendations based on nearest neighbor repositories.
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Any, Optional
from sklearn.metrics.pairwise import cosine_similarity

class ItemBasedKNNRecommender:
    def __init__(self, top_n_neighbors: int = 15):
        self.top_n_neighbors = top_n_neighbors
        self.user_to_idx: Dict[str, int] = {}
        self.idx_to_user: Dict[int, str] = {}
        self.repo_to_idx: Dict[str, int] = {}
        self.idx_to_repo: Dict[int, str] = {}
        self.interaction_matrix: Optional[np.ndarray] = None
        self.similarity_matrix: Optional[np.ndarray] = None
        self.repo_metadata: Dict[str, Dict[str, Any]] = {}

    def fit(self, df_train: pd.DataFrame, repo_metadata: Optional[List[Dict[str, Any]]] = None) -> "ItemBasedKNNRecommender":
        """
        Builds interaction matrix (users x repos) and computes repo-repo cosine similarities.
        """
        users = sorted(df_train["user"].unique())
        repos = sorted(df_train["repo"].unique())

        self.user_to_idx = {u: i for i, u in enumerate(users)}
        self.idx_to_user = {i: u for u, i in self.user_to_idx.items()}
        self.repo_to_idx = {r: i for i, r in enumerate(repos)}
        self.idx_to_repo = {i: r for r, i in self.repo_to_idx.items()}

        if repo_metadata:
            self.repo_metadata = {r["repo"]: r for r in repo_metadata}

        # Construct interaction matrix
        n_users = len(users)
        n_repos = len(repos)
        self.interaction_matrix = np.zeros((n_users, n_repos), dtype=np.float32)

        for _, row in df_train.iterrows():
            u_idx = self.user_to_idx[row["user"]]
            r_idx = self.repo_to_idx[row["repo"]]
            self.interaction_matrix[u_idx, r_idx] = float(row["score"])

        # Compute Item-Item cosine similarity matrix (repos x repos)
        # Transpose so rows are items
        item_matrix = self.interaction_matrix.T
        self.similarity_matrix = cosine_similarity(item_matrix)
        np.fill_diagonal(self.similarity_matrix, 0.0) # Zero out self-similarity
        return self

    def recommend(
        self,
        user: str,
        k: int = 10,
        filter_language: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Predicts ranking scores for uninteracted repositories for the given user.
        """
        if user not in self.user_to_idx:
            return [] # Cold-start user

        u_idx = self.user_to_idx[user]
        user_ratings = self.interaction_matrix[u_idx] # shape (n_repos,)
        interacted_repo_indices = np.where(user_ratings > 0)[0]

        if len(interacted_repo_indices) == 0:
            return []

        scores = {}
        explanations = {}

        n_repos = len(self.repo_to_idx)
        for cand_idx in range(n_repos):
            if cand_idx in interacted_repo_indices:
                continue # Skip already interacted repos

            cand_repo_name = self.idx_to_repo[cand_idx]
            cand_meta = self.repo_metadata.get(cand_repo_name, {})
            if filter_language and cand_meta.get("language", "").lower() != filter_language.lower():
                continue

            # Similarities between candidate and user's rated repos
            sims = self.similarity_matrix[cand_idx, interacted_repo_indices]
            ratings = user_ratings[interacted_repo_indices]

            # Top-N neighbors selection
            if len(sims) > self.top_n_neighbors:
                top_neighbor_idx = np.argsort(sims)[-self.top_n_neighbors:]
                sims = sims[top_neighbor_idx]
                ratings = ratings[top_neighbor_idx]

            pos_mask = sims > 0
            if not np.any(pos_mask):
                continue

            sim_sum = np.sum(sims[pos_mask])
            pred_score = float(np.sum(sims[pos_mask] * ratings[pos_mask]) / (sim_sum + 1e-8))

            # Find most influential past repo for explanation
            best_neighbor_local_idx = np.argmax(sims)
            best_past_repo_idx = interacted_repo_indices[best_neighbor_local_idx]
            best_past_repo_name = self.idx_to_repo[best_past_repo_idx]
            best_sim_score = float(sims[best_neighbor_local_idx])

            scores[cand_repo_name] = pred_score
            explanations[cand_repo_name] = (
                f"Collaborative match: developers who contributed to '{best_past_repo_name}' "
                f"also contributed here (similarity: {best_sim_score:.2f})."
            )

        # Sort candidate repos
        sorted_repos = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:k]
        results = []
        for repo_name, score in sorted_repos:
            meta = self.repo_metadata.get(repo_name, {})
            results.append({
                "repo": repo_name,
                "score": score,
                "language": meta.get("language", "Unknown"),
                "topics": meta.get("topics", []),
                "explanation": explanations.get(repo_name, "Collaborative similarity.")
            })

        return results
