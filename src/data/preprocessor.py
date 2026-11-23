"""
Data Preprocessor
Implements:
1. Implicit feedback weighting (merged_pr > opened_pr > issue_comment > opened_issue)
2. Sublinear log-damping: r_{u, i} = log(1 + Score(u, i))
3. Interaction filtering (min_user_interactions, min_repo_contributors)
4. Temporal / Leave-one-out train-test split
5. Negative sampling for implicit Neural RS training
"""

import math
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Any

DEFAULT_ACTION_WEIGHTS = {
    "merged_pr": 4.0,
    "opened_pr": 3.0,
    "issue_comment": 1.5,
    "opened_issue": 1.0
}

class GHDataPreprocessor:
    def __init__(
        self,
        action_weights: Dict[str, float] = None,
        min_user_interactions: int = 5,
        min_repo_contributors: int = 3
    ):
        self.action_weights = action_weights or DEFAULT_ACTION_WEIGHTS
        self.min_user_interactions = min_user_interactions
        self.min_repo_contributors = min_repo_contributors

    def compute_interaction_scores(self, events: List[Dict[str, Any]]) -> pd.DataFrame:
        """
        Calculates raw and log-damped implicit feedback scores for each user-repo pair.
        """
        df_events = pd.DataFrame(events)
        if df_events.empty:
            return pd.DataFrame(columns=["user", "repo", "raw_score", "score"])

        # Map action to numerical weight
        df_events["weight"] = df_events["action"].map(self.action_weights).fillna(1.0)

        # Aggregate by (user, repo)
        grouped = df_events.groupby(["user", "repo"])["weight"].sum().reset_index()
        grouped.rename(columns={"weight": "raw_score"}, inplace=True)

        # Apply log damping: r = log(1 + score)
        grouped["score"] = grouped["raw_score"].apply(lambda x: math.log1p(x))
        return grouped

    def filter_sparse_matrix(self, df_scores: pd.DataFrame) -> pd.DataFrame:
        """
        Iteratively prunes users and repositories below the threshold
        until stability is achieved.
        """
        df = df_scores.copy()
        while True:
            initial_len = len(df)
            user_counts = df.groupby("user")["repo"].nunique()
            valid_users = user_counts[user_counts >= self.min_user_interactions].index
            df = df[df["user"].isin(valid_users)]

            repo_counts = df.groupby("repo")["user"].nunique()
            valid_repos = repo_counts[repo_counts >= self.min_repo_contributors].index
            df = df[df["repo"].isin(valid_repos)]

            if len(df) == initial_len:
                break
        return df

    def create_train_test_split(
        self,
        df_scores: pd.DataFrame,
        test_ratio: float = 0.2,
        seed: int = 42
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Splits interactions into train and test sets on a per-user basis.
        Guarantees that test repos were interacted with by the user.
        """
        np.random.seed(seed)
        train_list = []
        test_list = []

        for user, user_df in df_scores.groupby("user"):
            n_items = len(user_df)
            if n_items < 2:
                train_list.append(user_df)
                continue
            
            n_test = max(1, int(round(n_items * test_ratio)))
            shuffled = user_df.sample(frac=1.0, random_state=seed)
            test_items = shuffled.iloc[:n_test]
            train_items = shuffled.iloc[n_test:]

            train_list.append(train_items)
            test_list.append(test_items)

        df_train = pd.concat(train_list, ignore_index=True)
        df_test = pd.concat(test_list, ignore_index=True)
        return df_train, df_test

    def generate_negative_samples(
        self,
        df_interactions: pd.DataFrame,
        all_repos: List[str],
        num_negatives: int = 4,
        seed: int = 42
    ) -> pd.DataFrame:
        """
        Generates negative samples (uninteracted repositories) for training implicit feedback models.
        Returns a DataFrame with columns: [user, repo, label] where positive=1, negative=0.
        """
        np.random.seed(seed)
        user_interacted = df_interactions.groupby("user")["repo"].apply(set).to_dict()
        all_repos_set = set(all_repos)

        rows = []
        for _, row in df_interactions.iterrows():
            user = row["user"]
            pos_repo = row["repo"]
            rows.append({"user": user, "repo": pos_repo, "label": 1.0, "score": row.get("score", 1.0)})

            interacted = user_interacted.get(user, set())
            neg_candidates = list(all_repos_set - interacted)
            if neg_candidates:
                chosen_negs = np.random.choice(
                    neg_candidates,
                    size=min(num_negatives, len(neg_candidates)),
                    replace=False
                )
                for neg_repo in chosen_negs:
                    rows.append({"user": user, "repo": neg_repo, "label": 0.0, "score": 0.0})

        return pd.DataFrame(rows)
