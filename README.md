# GitHub Issues Recommender System 🐙

A two-stage Recommender System that pairs open-source developers with GitHub repositories and beginner-friendly issues tailored to their skills and past contribution history.

Built for the **Recommender Systems** university course.

---

## 🌟 Key Features

1. **Two-Stage Recommender Architecture:**
   * **Stage 1 (Repository Candidate Generation):** Powered by a **Two-Tower Neural Network** encoding user interaction history and repository text semantics (READMEs & topics). Benchmarked against **Item-based kNN** and **Popularity** baselines.
   * **Stage 2 (Open Issue Semantic Ranking):** Evaluates real-time open issues within recommended repositories and ranks them based on semantic similarity to the developer's past work, boosting beginner tags (`good first issue`, `help wanted`).
2. **Transparent Explainability Layer:**
   * Traces each recommendation to specific developer experiences (nearest neighbor in latent space, overlapping tech stack, and shared tags).
3. **100% Free & Deployable:**
   * Optimized for zero-cost operation on **Hugging Face Spaces** (free 16 GB RAM / 2 vCPUs) or **Render**.

---

## 📁 Repository Structure

```
RS Project/
├── data/
│   ├── raw/                 # Raw GH Archive dumps & API logs
│   └── processed/           # Processed interaction matrices
├── src/
│   ├── data/
│   │   ├── collector.py     # GH Archive and GitHub API fetchers + mock generator
│   │   └── preprocessor.py  # Implicit feedback weights, log-damping, train/test split
│   ├── models/
│   │   ├── baseline_popularity.py   # Popularity baseline
│   │   ├── baseline_knn.py          # Item-based kNN collaborative filtering
│   │   ├── two_tower.py             # Two-Tower Neural Recommender
│   │   └── issue_ranker.py          # Stage 2 semantic issue ranking
│   ├── evaluation/
│   │   └── metrics.py       # Precision@K, Recall@K, NDCG@K, Catalog Coverage
│   └── app/
│       └── app.py           # Streamlit web application
├── tests/                   # Automated pipeline and model tests
├── requirements.txt         # Dependencies
├── Dockerfile               # Production container for free deployment
└── README.md
```

---

## 🚀 Quickstart Guide

### 1. Installation
Clone the repository and install dependencies:
```bash
git clone https://github.com/<your-username>/github-issues-recommender.git
cd github-issues-recommender
pip install -r requirements.txt
```

### 2. Run Test Suite & Benchmark
Run the automated end-to-end evaluation suite:
```bash
python -m pytest tests/test_evaluation_and_pipeline.py -s
```

### 3. Launch the Web App Locally
Run the interactive Streamlit dashboard:
```bash
streamlit run src/app/app.py
```
Open your browser at `http://localhost:8501`.

---

## 📊 Empirical Evaluation Benchmark (Live GitHub Data)

Evaluated on held-out interactions from real GitHub open-source repositories:

| Model | Precision@3 | Recall@3 | NDCG@3 | Precision@5 | Recall@5 | NDCG@5 | Recall@10 | NDCG@10 | Catalog Coverage |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Popularity Baseline** | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.7143 | 0.2145 | 61.1% |
| **Item-based kNN (CF)** | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0% *(Sparsity Collapse)* |
| **Two-Tower Neural RS** | **0.0952** | **0.2857** | **0.2143** | **0.1143** | **0.5714** | **0.3311** | **0.8571** | **0.4171** | **94.4%** |

### 🔬 Core Scientific Finding for Course Report:
* **Item-based kNN suffers from matrix sparsity collapse**: Real developers rarely have overlapping contributions across distinct repositories, leaving the collaborative similarity matrix disconnected.
* **Two-Tower Neural RS overcomes the sparsity problem**: By jointly embedding repository text semantics (topics, languages, and descriptions) and implicit interaction affinities in a shared latent space, it achieves **85.7% Recall@10** and **94.4% catalog coverage**, outperforming the non-personalized popularity baseline by nearly 2x on ranking quality (NDCG@10 of 0.4171 vs 0.2145).

---

## ☁️ Free Cloud Deployment (Hugging Face Spaces)

1. Create a free account at [huggingface.co](https://huggingface.co).
2. Click **New Space** $\to$ Select **Streamlit** (or **Docker**).
3. Push this repository's code to the Space's Git remote.
4. Your application will be live at `https://huggingface.co/spaces/<username>/<space-name>` with 16 GB RAM and 2 vCPUs at zero cost.
