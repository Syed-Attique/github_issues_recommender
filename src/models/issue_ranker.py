"""
Stage 2: Issue Semantic Ranker
Ranks open issues inside top recommended repositories
based on semantic match with the developer's past work and skills.
"""

from typing import List, Dict, Any, Optional
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

BEGINNER_LABELS = {"good first issue", "help wanted", "beginner-friendly", "easy", "up-for-grabs"}

class IssueSemanticRanker:
    def __init__(self, beginner_bonus: float = 0.15):
        self.beginner_bonus = beginner_bonus
        self.vectorizer = TfidfVectorizer(max_features=512, stop_words="english")

    def rank_issues_for_user(
        self,
        user_past_repos_metadata: List[Dict[str, Any]],
        candidate_issues: List[Dict[str, Any]],
        top_k: int = 5
    ) -> List[Dict[str, Any]]:
        """
        Ranks candidate open issues against the user's historical profile.
        """
        if not candidate_issues or not user_past_repos_metadata:
            return candidate_issues[:top_k]

        # 1. Build User Profile text from past repositories
        user_tokens = []
        for r in user_past_repos_metadata:
            lang = r.get("language", "")
            topics = " ".join(r.get("topics", []))
            desc = r.get("description", "")
            user_tokens.append(f"{lang} {topics} {desc}")

        user_text = " ".join(user_tokens)

        # 2. Build Issue texts
        issue_texts = []
        for iss in candidate_issues:
            title = iss.get("title", "")
            body = iss.get("body", "") or ""
            labels = " ".join(iss.get("labels", []))
            issue_texts.append(f"{title} {labels} {body[:300]}")

        # 3. Compute semantic vector similarity
        all_corpus = [user_text] + issue_texts
        tfidf_matrix = self.vectorizer.fit_transform(all_corpus)

        user_vec = tfidf_matrix[0:1]
        issue_vecs = tfidf_matrix[1:]

        sims = cosine_similarity(user_vec, issue_vecs).flatten()

        # 4. Score and explain
        ranked_results = []
        for idx, iss in enumerate(candidate_issues):
            raw_sim = float(sims[idx])
            labels = [l.lower() for l in iss.get("labels", [])]
            has_beginner_tag = any(b_label in labels for b_label in BEGINNER_LABELS)

            final_score = raw_sim + (self.beginner_bonus if has_beginner_tag else 0.0)

            # Generate explanation
            if has_beginner_tag and raw_sim > 0.1:
                explanation = f"High match ({final_score:.2f}) with your stack + labeled '{labels[0]}'."
            elif has_beginner_tag:
                explanation = f"Friendly beginner issue: labeled '{labels[0]}'."
            else:
                explanation = f"Semantic skill match ({final_score:.2f}) based on your past contributions."

            ranked_results.append({
                "number": iss.get("number"),
                "title": iss.get("title"),
                "html_url": iss.get("html_url"),
                "labels": iss.get("labels", []),
                "repo": iss.get("repo"),
                "score": round(final_score, 4),
                "explanation": explanation
            })

        # Sort descending by final score
        ranked_results.sort(key=lambda x: x["score"], reverse=True)
        return ranked_results[:top_k]
