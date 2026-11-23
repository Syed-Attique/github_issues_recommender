"""
GitHub Data Collector
Handles fetching from:
1. GH Archive (public hourly event dumps)
2. GitHub REST API (repository metadata, READMEs, open issues)
3. Synthetic / Mock dataset generator for rapid offline prototyping
"""

import os
import json
import gzip
import time
import random
from typing import Dict, List, Any, Optional
import requests

KNOWN_BOTS = {
    "dependabot[bot]", "renovate[bot]", "github-actions[bot]",
    "greenkeeper[bot]", "snyk-bot", "codecov[bot]", "stale[bot]"
}

class GHDataCollector:
    def __init__(self, github_token: Optional[str] = None):
        self.github_token = github_token or os.getenv("GITHUB_TOKEN")
        self.session = requests.Session()
        if self.github_token:
            self.session.headers.update({
                "Authorization": f"Bearer {self.github_token}",
                "Accept": "application/vnd.github.v3+json"
            })

    def download_gh_archive_hour(self, year: int, month: int, day: int, hour: int, output_dir: str = "data/raw") -> str:
        """Downloads an hourly archive from GH Archive (e.g. 2026-01-01-15.json.gz)."""
        os.makedirs(output_dir, exist_ok=True)
        filename = f"{year:04d}-{month:02d}-{day:02d}-{hour}.json.gz"
        url = f"https://data.gharchive.org/{filename}"
        target_path = os.path.join(output_dir, filename)

        if os.path.exists(target_path):
            return target_path

        response = requests.get(url, stream=True)
        response.raise_for_status()
        with open(target_path, "wb") as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        return target_path

    def parse_archive_events(self, file_path: str) -> List[Dict[str, Any]]:
        """Parses interaction events relevant to open-source contributions."""
        events = []
        with gzip.open(file_path, "rt", encoding="utf-8", errors="ignore") as f:
            for line in f:
                try:
                    payload = json.loads(line)
                    event_type = payload.get("type")
                    actor = payload.get("actor", {}).get("login")
                    repo_name = payload.get("repo", {}).get("name")
                    created_at = payload.get("created_at")

                    if not actor or not repo_name or actor.lower() in KNOWN_BOTS:
                        continue

                    action_type = None
                    if event_type == "PullRequestEvent":
                        pr_action = payload.get("payload", {}).get("action")
                        is_merged = payload.get("payload", {}).get("pull_request", {}).get("merged", False)
                        if pr_action == "closed" and is_merged:
                            action_type = "merged_pr"
                        elif pr_action == "opened":
                            action_type = "opened_pr"
                    elif event_type == "IssueCommentEvent" and payload.get("payload", {}).get("action") == "created":
                        action_type = "issue_comment"
                    elif event_type == "IssuesEvent" and payload.get("payload", {}).get("action") == "opened":
                        action_type = "opened_issue"

                    if action_type:
                        events.append({
                            "user": actor,
                            "repo": repo_name,
                            "action": action_type,
                            "created_at": created_at
                        })
                except Exception:
                    continue
        return events

    def fetch_repo_metadata(self, repo_full_name: str) -> Optional[Dict[str, Any]]:
        """Fetches repo metadata (language, topics, description, stars) from GitHub REST API."""
        url = f"https://api.github.com/repos/{repo_full_name}"
        try:
            resp = self.session.get(url, timeout=10)
            if resp.status_code == 200:
                data = resp.json()
                return {
                    "repo": repo_full_name,
                    "language": data.get("language") or "Unknown",
                    "topics": data.get("topics", []),
                    "description": data.get("description") or "",
                    "stars": data.get("stargazers_count", 0),
                    "forks": data.get("forks_count", 0)
                }
        except Exception as e:
            print(f"Error fetching metadata for {repo_full_name}: {e}")
        return None

    def fetch_open_issues(self, repo_full_name: str, max_issues: int = 10) -> List[Dict[str, Any]]:
        """Fetches beginner-friendly open issues from GitHub REST API."""
        url = f"https://api.github.com/repos/{repo_full_name}/issues"
        params = {"state": "open", "per_page": max_issues}
        results = []
        try:
            resp = self.session.get(url, params=params, timeout=10)
            if resp.status_code == 200:
                for item in resp.json():
                    # GitHub API returns pull requests in issues endpoint if not filtered
                    if "pull_request" in item:
                        continue
                    labels = [l.get("name", "") for l in item.get("labels", [])]
                    results.append({
                        "issue_id": item.get("id"),
                        "number": item.get("number"),
                        "title": item.get("title", ""),
                        "body": item.get("body", "") or "",
                        "html_url": item.get("html_url"),
                        "labels": labels,
                        "created_at": item.get("created_at")
                    })
        except Exception as e:
            print(f"Error fetching issues for {repo_full_name}: {e}")
        return results

    @staticmethod
    def generate_mock_dataset(
        num_users: int = 150,
        num_repos: int = 40,
        num_events: int = 2500,
        seed: int = 42
    ) -> Dict[str, Any]:
        """
        Generates realistic synthetic data for offline testing & rapid development.
        Produces consistent user activity, repository metadata, and open issues.
        """
        random.seed(seed)
        languages = ["Python", "TypeScript", "JavaScript", "Go", "Rust", "C++", "Java"]
        topics_pool = [
            "machine-learning", "deep-learning", "recommender-systems", "fastapi",
            "react", "web", "async", "database", "cli", "compiler", "api", "docker",
            "microservices", "frontend", "backend", "cloud"
        ]

        # Generate Repositories
        repos = []
        for i in range(num_repos):
            lang = random.choice(languages)
            repo_topics = random.sample(topics_pool, k=random.randint(2, 5))
            repo_name = f"org_{i % 10}/project-{lang.lower()}-{i}"
            repos.append({
                "repo": repo_name,
                "language": lang,
                "topics": repo_topics,
                "description": f"A high performance {lang} framework for {', '.join(repo_topics)}.",
                "stars": random.randint(50, 15000),
                "readme_summary": f"Welcome to {repo_name}. This project is built using {lang}. It provides toolkits for {', '.join(repo_topics)}."
            })

        # Generate Users with language affinities
        users = [f"developer_{u}" for u in range(num_users)]
        user_affinities = {
            u: random.sample(languages, k=random.randint(1, 3)) for u in users
        }

        # Generate Events
        action_types = ["merged_pr", "opened_pr", "issue_comment", "opened_issue"]
        action_weights = [0.15, 0.25, 0.40, 0.20]

        events = []
        for _ in range(num_events):
            user = random.choice(users)
            user_langs = user_affinities[user]
            # Bias repo selection toward user's preferred languages
            candidate_repos = [r for r in repos if r["language"] in user_langs]
            if not candidate_repos or random.random() < 0.2:
                candidate_repos = repos
            chosen_repo = random.choice(candidate_repos)["repo"]
            action = random.choices(action_types, weights=action_weights, k=1)[0]
            events.append({
                "user": user,
                "repo": chosen_repo,
                "action": action,
                "created_at": "2026-01-15T12:00:00Z"
            })

        # Generate Open Issues for each repo
        issues_pool = [
            ("Fix asynchronous event loop blocking in worker thread", "The async worker blocks when processing large batches. Needs refactoring with asyncio/multiprocessing.", ["good first issue", "bug"]),
            ("Add support for custom evaluation metrics in pipeline", "Implement NDCG and Precision metrics for the ranking module.", ["help wanted", "enhancement"]),
            ("Update README documentation and quickstart examples", "Documentation is missing docker run instructions.", ["documentation", "good first issue"]),
            ("Optimize memory footprint in high throughput stream", "Investigate memory leak occurring during continuous message ingestion.", ["bug", "performance"]),
            ("Create unit tests for API endpoints", "Add pytest coverage for auth and repository recommendation endpoints.", ["good first issue", "testing"]),
            ("Add TypeScript types for API client response", "Ensure strict typings for payload responses and error objects.", ["frontend", "good first issue"])
        ]

        open_issues = []
        for r in repos:
            num_repo_issues = random.randint(2, 5)
            for j in range(num_repo_issues):
                sample_issue = random.choice(issues_pool)
                open_issues.append({
                    "repo": r["repo"],
                    "number": j + 1,
                    "title": f"[{r['language']}] {sample_issue[0]}",
                    "body": sample_issue[1],
                    "labels": sample_issue[2],
                    "html_url": f"https://github.com/{r['repo']}/issues/{j + 1}"
                })

        return {
            "events": events,
            "repos": repos,
            "open_issues": open_issues
        }
