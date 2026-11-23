# Github Issues Recommender System - Project Proposal
**Course:** Recommender Systems  
**Project Track:** Deep Learning & Hybrid Recommender Systems  

### Group Members
1. Mustafa Hussain Khan, 23K-0912  
2. Hasnain Kazmi, 23K-0692  
3. Shariq Hassan Khan, 23K-0626  
4. Syed Attique Ur Rehman, 23K-0714  

---

### 1. Introduction
Finding a good first open-source contribution is challenging. A developer typically has to browse thousands of repositories, sift through noisy issue trackers, and guess which problems match their current skills. Keyword search and the standard `"good first issue"` label help, but they remain static and oblivious to an individual developer's past contribution history and technical competencies.

We propose a deployed, web-based recommender system where a user enters their GitHub username and receives a personalized, ranked list of repositories and actively open issues they are likely to contribute to. Each recommendation is paired with an intuitive explanation (e.g., *"Recommended because your Python & asynchronous programming background matches this repository's core stack and open issues"*).

To evaluate performance rigorously, our system compares classic collaborative filtering (Item-based kNN) against a modern **Two-Tower Neural Network Recommender System** that learns joint user-repository representations from both implicit interaction history and rich semantic metadata.

---

### 2. Dataset & Free Data Pipeline
The system relies exclusively on free, public data sources:
* **GH Archive:** Public, hourly event dumps of global GitHub activity. We extract pull request and issue interaction events over a 1-to-3 month window to construct the implicit user-repository interaction matrix.
* **GitHub REST API (Free Tier with Personal Access Token):** Enriches repositories with metadata including topics, primary language, star counts, README summaries, and current open issues labeled with tags such as `"good first issue"`, `"help wanted"`, or `"beginner-friendly"`.

**Data Cleaning & Sparsity Control:**
* Filter out automated bot accounts (e.g., Dependabot, Renovate) and repository forks.
* Retain users with at least 5 meaningful contributions and repositories with enough distinct contributors to maintain matrix density.
* Target dataset size: Several thousand clean developer profiles and repositories.
* The collection scripts, preprocessing notebooks, and cleaned dataset will be publicly hosted on GitHub for complete reproducibility.

---

### 3. Ratings & Implicit Feedback Formulation
GitHub lacks explicit star ratings for task suitability, so we model developer preferences using weighted implicit feedback. A user-repository interaction score is derived from progressive engagement weights:
$$\text{Score}(u, i) = w_1 \cdot (\text{Merged PR}) + w_2 \cdot (\text{Opened PR}) + w_3 \cdot (\text{Issue Comment}) + w_4 \cdot (\text{Issue Opened})$$
*(where $w_1 > w_2 > w_3 > w_4$).*

To prevent power-law distortions where extreme contributors dominate the feature space, a sub-linear log-damping transformation is applied:
$$r_{u, i} = \log(1 + \text{Score}(u, i))$$

---

### 4. Two-Stage Recommendation Architecture
Due to the short lifespan of GitHub issues, recommending individual issues directly via collaborative filtering suffers from extreme sparsity. We decouple the problem into a two-stage pipeline:

* **Stage 1 (Repository Recommendation):** Candidate generation and ranking of repositories using user interaction history and repository domain profiles.
* **Stage 2 (Issue Semantic Ranking):** Within the top recommended repositories, currently open issues are fetched in real-time via the GitHub API and re-ranked using neural semantic similarity between the issue's title/description and the developer's encoded skill vector.

---

### 5. Recommendation Techniques
We will implement and benchmark three distinct models:

1. **Popularity Baseline:** Recommends the globally most-contributed repositories within the target language/topics. Serves as the minimal benchmark that all personalized models must outperform.
2. **Item-Based kNN (Collaborative Baseline):** Computes cosine similarity across repositories based on the user-repository implicit interaction matrix. Generates recommendations based on repositories co-contributed by similar developers.
3. **Two-Tower Neural Network Recommender (Neural RS - Primary Model):**
   * **User Tower:** Encodes the developer's past interaction history, language distribution, and topic affinities into a dense latent representation vector $\mathbf{u} \in \mathbb{R}^d$.
   * **Repository Tower:** Encodes repository metadata by passing README text and issue descriptions through a lightweight, open-source pre-trained Sentence Transformer (`all-MiniLM-L6-v2`) combined with categorical embeddings for primary languages and topics into a dense vector $\mathbf{v} \in \mathbb{R}^d$.
   * **Matching & Scoring:** The predicted preference is computed via the dot product/cosine similarity: $\hat{y}_{u, v} = \langle \mathbf{u}, \mathbf{v} \rangle$, optimized using Binary Cross-Entropy with negative sampling or Bayesian Personalized Ranking (BPR) loss.
4. **Weighted Hybrid (Optional Ensemble):** Blends normalized scores from the Item-based kNN and the Two-Tower Neural Network, with the blending weight tuned on a held-out validation split. A cold-start fallback relies entirely on the repository tower embeddings when a new user enters the system.
5. **Explainability Layer:** Transparent explanations are generated by:
   * Identifying the nearest previously-contributed repository in the latent embedding space.
   * Highlighting overlapping skill tags, programming languages, and matching semantic keywords extracted from the issue and README embeddings.

---

### 6. Evaluation Plan
* **Offline Quantitative Metrics:**
  * Each user's contributions are split temporally; recent interactions are held out for testing.
  * Evaluated on standard ranking metrics: **Precision@K**, **Recall@K**, and **NDCG@K** (for $K \in \{5, 10\}$).
  * **Catalog Coverage & Gini Diversity Index:** To verify that the neural model recommends long-tail repositories rather than collapsing into a popularity bias.
* **Online & Qualitative Human Judgement:**
  * The deployed application will feature a binary feedback mechanism (thumbs up / thumbs down) for both recommendations and explanation relevance.
  * A blind evaluation survey distributed to peer student developers to score recommendation relevance and explanation clarity on a 1-to-5 Likert scale.

---

### 7. Deployment & Zero-Cost Architecture Plan
The entire system is intentionally designed to be trained and hosted with zero financial cost:
* **Training (Free Tier):** Google Colab (free T4 GPU / CPU) or local machine. Using a lightweight architecture and pre-computed embeddings, model training will take under 10 minutes.
* **Vector Inference:** Repository embeddings are pre-computed offline. At runtime, vector search is performed in memory using FAISS / NumPy cosine similarity, running in under 30 ms on CPU.
* **Web Application:** Built with **Streamlit** or **FastAPI + lightweight frontend**.
* **Hosting (100% Free Tier):** Deployed via Docker or direct Git sync on **Hugging Face Spaces** (providing 2 vCPUs and 16 GB RAM for free) or **Render Free Web Services**.
* **Public Codebase:** All code, data collection scripts, trained model checkpoints, and evaluation notebooks will be hosted in a public GitHub repository with comprehensive setup instructions.

---

### 8. Undertaking
We pledge not to use any dishonest means at any stage of this project. All data collection, model design, implementation, and documentation will be our own original work, and any third-party datasets, open-source libraries, or pre-trained models will be cited accurately.
