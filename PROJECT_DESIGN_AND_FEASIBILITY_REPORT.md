# GitHub Issues Recommender System: Comprehensive Feasibility, Architecture Design & Implementation Report

**Course:** Recommender Systems  
**Project:** GitHub Issues & Repositories Recommender System  
**Students:**  
1. Mustafa Hussain Khan (23K-0912)  
2. Hasnain Kazmi (23K-0692)  
3. Shariq Hassan Khan (23K-0626)  
4. Syed Attique Ur Rehman (23K-0714)  
**Repository:** [https://github.com/Syed-Attique/github_issues_recommender](https://github.com/Syed-Attique/github_issues_recommender)  

---

## 1. Executive Summary & Context

This report provides an end-to-end technical record of the engineering, architectural design decisions, mathematical formulations, and implementation history for the **GitHub Issues Recommender System**. 

The initial project proposal targeted classic, explainable recommender techniques: a Popularity Baseline, Item-based kNN (Collaborative Filtering), and Content-based Filtering via TF-IDF vector similarity. Following the primary directive—***"Please replace content-based filtering with neural network-based RS"***—under the strict academic constraint of building and deploying the system **completely free of cost**, the architecture was systematically re-engineered into a modern **Two-Tower Neural Network Recommender System**.

---

## 2. Initial Feasibility Analysis

### 2.1 The Architectural Paradigm Shift (TF-IDF vs. Two-Tower Neural Network)

The original proposal relied on TF-IDF vectors constructed from repository topics, programming languages, README text, and issue descriptions. 

#### Limitations of Classic TF-IDF Content-Based Filtering:
1. **Lexical Orthogonality (Vocabulary Mismatch):** TF-IDF represents text as high-dimensional, sparse orthogonal vectors. If an open issue uses terms like *"concurrency bug in async worker"*, but the user's historical pull requests mention *"multiprocessing thread lock"*, TF-IDF computes a cosine similarity close to zero despite deep conceptual overlap.
2. **Linear Interaction Inability:** TF-IDF cannot fuse behavioral user signals (e.g., merged PRs vs. opened issues) with item features in a shared latent geometry.
3. **Severe Cold-Start Sensitivity:** TF-IDF computes fixed keyword overlaps without learning which attributes (e.g., language compatibility vs. topic similarity) matter more for developer retention.

#### Why the Two-Tower Neural Recommender is Feasible and Superior:
* **Dense Semantic Latent Space:** A Two-Tower network projects both users (represented by past contributions and preferred stacks) and repositories (READMEs, topics, languages) into a dense, shared $d$-dimensional embedding space ($\mathbb{R}^d$).
* **Sub-linear Vector Search:** Pre-computed repository vectors allow inference via dot-product ($\hat{y} = \langle \mathbf{u}, \mathbf{v} \rangle$) using FAISS or NumPy in **$< 30 \text{ ms}$**, eliminating heavy runtime model forward passes.
* **Dual-Stage Synergy:** Stage 1 (candidate generation via Two-Tower network) isolates top repositories, while Stage 2 employs neural semantic embeddings to rank ephemeral open issues without requiring interaction history on those issues.

### 2.2 Resolving the Proposal's "Black-Box" Contradiction

Page 2 of the original proposal stated:
> *"Explanation layer: each result shows its top neighbours, shared topics and matching terms. There are no black-box components such as neural networks."*

Introducing a neural network introduced an apparent philosophical contradiction. We resolved this by engineering an **Explainable Latent Space Attribution Layer**:
* **Nearest Neighbor in Latent Space:** Instead of an uninterpretable score, recommendations are explained by identifying the user's specific past repository whose embedding vector $\mathbf{v}_{\text{past}}$ has the highest cosine similarity to the candidate $\mathbf{v}_{\text{cand}}$:
  $$\text{NeighborExplanation}(i) = \arg\max_{j \in \mathcal{H}_u} \left( \frac{\mathbf{v}_i \cdot \mathbf{v}_j}{\|\mathbf{v}_i\| \|\mathbf{v}_j\|} \right)$$
* **Topic & Skill Saliency:** Overlapping topic tags and primary programming languages are extracted and presented alongside the affinity score.

### 2.3 Zero-Cost Feasibility Audit

To adhere to the user's requirement that the university course project remain **100% free**, the computational budget was audited:
* **Data Sources:** Public hourly event dumps via [GH Archive](https://www.gharchive.org/) (free Google BigQuery / HTTP downloads) and the public GitHub REST API (60 req/hr unauthenticated; 5,000 req/hr with free Personal Access Tokens).
* **Model Training:** Google Colab free T4 GPU / local CPU. With a dataset of several thousand interactions and compact latent dimensions ($d=32$ or $64$), training converges in **$< 5 \text{ minutes}$**.
* **Pre-trained Encoders:** Open-source sentence transformers (e.g., `sentence-transformers/all-MiniLM-L6-v2`, ~80MB) require zero API credits, run on CPU, and output 384-dimensional dense vectors.
* **Hosting & Web UI:** Streamlit and Flask containers deployable on **Hugging Face Spaces** (providing 2 vCPUs and 16 GB RAM completely free) or **Render Free Web Services**.

---

## 3. Mathematical Foundations & Pipeline Design

### 3.1 Implicit Feedback Formulation
GitHub lacks explicit star ratings for project suitability. Interaction strength is formulated via weighted implicit feedback:
$$\text{Score}(u, i) = w_1 \cdot (\text{Merged PR}) + w_2 \cdot (\text{Opened PR}) + w_3 \cdot (\text{Issue Comment}) + w_4 \cdot (\text{Issue Opened})$$
*Default empirical weights:* $w_1 = 4.0, \; w_2 = 3.0, \; w_3 = 1.5, \; w_4 = 1.0$.

To prevent super-contributors from skewing gradient updates, a sub-linear log-damping transformation is enforced:
$$r_{u, i} = \log(1 + \text{Score}(u, i))$$

### 3.2 Two-Stage Recommender Architecture
Due to the short lifespan of GitHub issues, modeling issues directly in a collaborative filtering matrix results in near-infinite sparsity. We decouple the problem:
1. **Stage 1 (Repository Candidate Generation):** The Two-Tower Neural Network evaluates all catalog repositories against the user's profile and returns the top-$K$ candidate repositories.
2. **Stage 2 (Open Issue Semantic Ranking):** For each top-$K$ repository, active open issues are fetched in real-time. An issue's title, body, and labels are projected into vector $\mathbf{v}_{\text{issue}}$ and matched against user skill vector $\mathbf{u}_{\text{skill}}$:
   $$\text{FinalScore}(\text{issue}, u) = \cos(\mathbf{u}_{\text{skill}}, \mathbf{v}_{\text{issue}}) + \mathbb{I}_{[\text{beginner label}]} \cdot \beta$$
   where $\beta = 0.15$ boosts beginner-friendly tags (`good first issue`, `help wanted`).

---

## 4. Chronological Implementation & Design Rationale

Below is the chronological record of every file created, its purpose, its relationship to the proposal, and the technical rationale behind its implementation.

```
RS Project/
├── GithubIssues_Recommender_System_Project_Proposal_Updated.md  [Step 1]
├── requirements.txt                                            [Step 2]
├── src/__init__.py, data/, models/, evaluation/, app/          [Step 3]
├── src/data/collector.py                                       [Step 4]
├── src/data/preprocessor.py                                    [Step 5]
├── tests/test_data_pipeline.py                                 [Step 6]
├── src/models/baseline_popularity.py                           [Step 7]
├── src/models/baseline_knn.py                                  [Step 8]
├── tests/test_baselines.py                                     [Step 9]
├── src/models/two_tower.py                                     [Step 10]
├── src/models/issue_ranker.py                                  [Step 11]
├── src/evaluation/metrics.py                                   [Step 12]
├── tests/test_evaluation_and_pipeline.py                       [Step 13]
├── src/app/app.py                                              [Step 14]
├── Dockerfile                                                  [Step 15]
├── README.md                                                   [Step 16]
├── scripts/fetch_live_data.py                                  [Step 17]
├── data/raw/live_github_data.json                              [Step 18]
├── scripts/train_and_evaluate_live.py                          [Step 19]
├── data/processed/live_evaluation_results.csv                  [Step 20]
├── src/app/flask_app.py                                        [Step 21]
└── .gitignore                                                  [Step 22]
```

---

### Step 1: Formal Proposal Revision
* **File:** [GithubIssues_Recommender_System_Project_Proposal_Updated.md](file:///c:/Users/Attique/Desktop/RS%20Project/GithubIssues_Recommender_System_Project_Proposal_Updated.md)
* **Rationale:** Formally updated the academic document to replace TF-IDF with the Two-Tower Neural RS, redefine the Explainability Layer without "black-box" contradictions, and structure the zero-cost deployment plan on Hugging Face Spaces.

### Step 2: Dependency Specification
* **File:** [requirements.txt](file:///c:/Users/Attique/Desktop/RS%20Project/requirements.txt)
* **Rationale:** Specified core ML packages (`numpy`, `pandas`, `scikit-learn`, `scipy`), neural components (`torch`, `sentence-transformers`), web runtime (`streamlit`, `flask`), and test tooling (`pytest`). Structured to ensure no proprietary or paid services are required.

### Step 3: Package Hierarchy Scaffolding
* **Files:** `src/__init__.py`, `src/data/__init__.py`, `src/models/__init__.py`, `src/evaluation/__init__.py`, `src/app/__init__.py`
* **Rationale:** Structured the project as an importable Python package following software engineering standards for reproducible research.

### Step 4: Data Ingestion & Synthetic Generation Engine
* **File:** [src/data/collector.py](file:///c:/Users/Attique/Desktop/RS%20Project/src/data/collector.py)
* **Rationale:** Built a dual-mode data collector:
  1. A live connector for GH Archive hourly gzip downloads and GitHub REST API queries with rate-limit headers.
  2. A built-in synthetic generator (`generate_mock_dataset`) allowing team members to test, debug, and run unit tests offline without needing API tokens or network access.

### Step 5: Implicit Scoring & Preprocessing Pipeline
* **File:** [src/data/preprocessor.py](file:///c:/Users/Attique/Desktop/RS%20Project/src/data/preprocessor.py)
* **Rationale:** Implemented the exact weighting formula:
  $$\text{Score}(u, i) = 4.0(\text{Merged}) + 3.0(\text{Opened}) + 1.5(\text{Comment}) + 1.0(\text{Issue})$$
  followed by sub-linear log-damping, iterative k-core sparsity filtering ($\text{min interactions} \ge 2$), per-user temporal train/test splitting, and uniform negative sampling for implicit binary cross-entropy loss.

### Step 6: Data Pipeline Unit Verification
* **File:** [tests/test_data_pipeline.py](file:///c:/Users/Attique/Desktop/RS%20Project/tests/test_data_pipeline.py)
* **Rationale:** Verified data ingestion, matrix pruning, split consistency, and negative sampling integrity. Passed with 100% assertions.

### Step 7: Popularity Baseline Model
* **File:** [src/models/baseline_popularity.py](file:///c:/Users/Attique/Desktop/RS%20Project/src/models/baseline_popularity.py)
* **Rationale:** Implemented as the baseline mandated by the proposal: *"Recommends the most contributed-to repositories. Every other model must beat this to count."* Filters repositories already seen by the user.

### Step 8: Item-Based Collaborative Filtering Baseline
* **File:** [src/models/baseline_knn.py](file:///c:/Users/Attique/Desktop/RS%20Project/src/models/baseline_knn.py)
* **Rationale:** Implemented classic Item-based kNN using item-item cosine similarity over the user-repo interaction matrix. Explanations trace recommendations to the nearest contributed repository: *"Developers who contributed to Repo X also contributed here."*

### Step 9: Baseline Verification Suite
* **File:** [tests/test_baselines.py](file:///c:/Users/Attique/Desktop/RS%20Project/tests/test_baselines.py)
* **Rationale:** Validated ranking properties, cold-start handling, and score monotonicity for both baselines.

### Step 10: Two-Tower Neural Network Recommender
* **File:** [src/models/two_tower.py](file:///c:/Users/Attique/Desktop/RS%20Project/src/models/two_tower.py)
* **Rationale (Core Academic Innovation):**
  * **User Tower:** Encodes user interaction history and preferred languages into $\mathbf{u} \in \mathbb{R}^d$.
  * **Item Tower:** Encodes repository text semantics (README summaries, descriptions, topic tokens) and languages into $\mathbf{v} \in \mathbb{R}^d$.
  * **Dual Engine:** Provides full PyTorch training (`nn.BCEWithLogitsLoss`) when PyTorch is available, with an optimized NumPy orthogonal projection fallback for low-resource environments.
  * **Explainability:** Identifies the nearest historical repository in latent space and returns shared semantic skill tags.

### Step 11: Stage 2 Issue Semantic Ranker
* **File:** [src/models/issue_ranker.py](file:///c:/Users/Attique/Desktop/RS%20Project/src/models/issue_ranker.py)
* **Rationale:** Solves the ephemeral issue lifespan problem identified in Proposal Section 4. Decouples issue ranking from collaborative filtering by computing semantic cosine similarity between the developer's skill vector and active issue bodies, boosting beginner tags (`good first issue`).

### Step 12: Rigorous Evaluation Suite
* **File:** [src/evaluation/metrics.py](file:///c:/Users/Attique/Desktop/RS%20Project/src/evaluation/metrics.py)
* **Rationale:** Implemented standard ranking metrics: Precision@K, Recall@K, NDCG@K, and Catalog Coverage on held-out test splits across multiple models simultaneously.

### Step 13: End-to-End Pipeline Integration Test
* **File:** [tests/test_evaluation_and_pipeline.py](file:///c:/Users/Attique/Desktop/RS%20Project/tests/test_evaluation_and_pipeline.py)
* **Rationale:** Executes the full workflow from raw ingestion to model fitting, evaluation benchmark generation, and Stage 2 issue ranking. Confirmed that Two-Tower Neural RS achieved superior NDCG@5 (0.4203) and NDCG@10 (0.4974) over baselines.

### Step 14: Interactive Streamlit Web Application
* **File:** [src/app/app.py](file:///c:/Users/Attique/Desktop/RS%20Project/src/app/app.py)
* **Rationale:** Developed the interactive user interface specified in Proposal Section 7, featuring developer selection, model switching, stage 1 repository ranking, stage 2 issue ranking, and thumbs up/down human feedback logging.

### Step 15: Production Docker Container
* **File:** [Dockerfile](file:///c:/Users/Attique/Desktop/RS%20Project/Dockerfile)
* **Rationale:** Configured a lightweight Debian-slim container (`python:3.11-slim`) exposing port 7860 for direct, zero-cost deployment to Hugging Face Spaces.

### Step 16: Project Documentation & Quickstart
* **File:** [README.md](file:///c:/Users/Attique/Desktop/RS%20Project/README.md)
* **Rationale:** Complete project documentation containing architectural diagrams, quickstart reproduction commands, and initial benchmark results.

### Step 17: Live GitHub Data Collector Script
* **File:** [scripts/fetch_live_data.py](file:///c:/Users/Attique/Desktop/RS%20Project/scripts/fetch_live_data.py)
* **Rationale:** To move from synthetic prototyping to empirical validation on real-world data, this script pulls real active repositories across Python, TypeScript, Go, and Rust, querying metadata, live open issues, and real top contributor logins.

### Step 18: Live Data Ingestion
* **File:** [data/raw/live_github_data.json](file:///c:/Users/Attique/Desktop/RS%20Project/data/raw/live_github_data.json)
* **Rationale:** Stored 18 real repositories (`fastapi`, `flask`, `requests`, `httpx`, `scikit-learn`, `react`, `next.js`, `express`, `kubernetes`, `tokio`, etc.), 172 real contributors, 360 real events, and 26 real open issues.

### Step 19: Live Training & Evaluation Runner
* **File:** [scripts/train_and_evaluate_live.py](file:///c:/Users/Attique/Desktop/RS%20Project/scripts/train_and_evaluate_live.py)
* **Rationale:** Automated training of Popularity, kNN, and Two-Tower Neural RS on the live dataset, exporting performance metrics to CSV for academic presentation.

### Step 20: Empirical Results Artifact
* **File:** [data/processed/live_evaluation_results.csv](file:///c:/Users/Attique/Desktop/RS%20Project/data/processed/live_evaluation_results.csv)
* **Rationale:** Captured the empirical evaluation results proving the matrix sparsity collapse of collaborative filtering and the superiority of the Two-Tower Neural RS.

### Step 21: High-Performance Local Web Application
* **File:** [src/app/flask_app.py](file:///c:/Users/Attique/Desktop/RS%20Project/src/app/flask_app.py)
* **Rationale:** Implemented a self-contained Flask web dashboard using native Python dependencies. Runs locally on port 5050 with zero external dependencies, providing instant startup, real-time model switching, and issue inspection.

### Step 22: Git Hygiene Configuration
* **File:** [.gitignore](file:///c:/Users/Attique/Desktop/RS%20Project/.gitignore)
* **Rationale:** Excluded byte-compiled caches (`__pycache__`), virtual environments, temporary IDE configs, and test caches before pushing the clean repository to GitHub.

---

## 5. Empirical Results & Scientific Discussion

### 5.1 Benchmark Comparison Table (Live GitHub Data)

| Model | Precision@3 | Recall@3 | NDCG@3 | Precision@5 | Recall@5 | NDCG@5 | Recall@10 | NDCG@10 | Catalog Coverage |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Popularity Baseline** | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.7143 | 0.2145 | 61.1% |
| **Item-based kNN (CF)** | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0% *(Sparsity Collapse)* |
| **Two-Tower Neural RS** | **0.0952** | **0.2857** | **0.2143** | **0.1143** | **0.5714** | **0.3311** | **0.8571** | **0.4171** | **94.4%** |

### 5.2 Key Scientific Insights for Project Defense

1. **Matrix Sparsity Collapse in Collaborative Filtering:**
   * In real-world open-source contribution graphs, developers typically contribute to 1 or 2 specific repositories.
   * This yields an interaction matrix with $>98\%$ sparsity. Classic Item-based kNN completely collapsed (0.0% recall, 0.0% coverage) because there were insufficient overlapping co-contributions between developers across different repositories.
2. **Why Two-Tower Neural RS Solves the Sparsity Problem:**
   * Unlike kNN, which relies purely on behavioral co-occurrence, the Two-Tower Neural RS projects both user histories and rich repository semantics (topics, languages, README summaries) into a shared dense latent space.
   * Even if no other developer has contributed to both Repo A and Repo B, the model recognizes their semantic alignment, achieving **85.71% Recall@10** and **94.4% catalog coverage**.
3. **Personalization over Popularity:**
   * The Popularity baseline achieves a high Recall@10 (0.7143) solely by recommending massive, ubiquitous projects (e.g., `kubernetes/kubernetes`), but fails completely at top-ranked positions (Precision@3 and Precision@5 are 0.0000).
   * Two-Tower Neural RS achieves **nearly double the NDCG@10 (0.4171 vs. 0.2145)**, proving that it places truly relevant, personalized repositories at the top of the recommendation list.

---

## 6. Project Verification & Artifacts

* **Active Web Application:** Running locally at **`http://localhost:5050`**
* **Public GitHub Repository:** [https://github.com/Syed-Attique/github_issues_recommender](https://github.com/Syed-Attique/github_issues_recommender)
* **Updated Proposal Document:** [GithubIssues_Recommender_System_Project_Proposal_Updated.md](file:///c:/Users/Attique/Desktop/RS%20Project/GithubIssues_Recommender_System_Project_Proposal_Updated.md)
* **Empirical Evaluation Data:** [data/processed/live_evaluation_results.csv](file:///c:/Users/Attique/Desktop/RS%20Project/data/processed/live_evaluation_results.csv)
