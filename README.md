# Non-Binary Hierarchical Clustering

This project performs Non-Binary hierarchical clustering on arXiv and Wikipedia dataset.

---

## 🔧 Setup Instructions

### 1. Prepare the Dataset
Download and add the following dataset to the `resources/` folder:

- **File:** `arxiv-metadata-oai-snapshot.json`  
- **Source:** [Kaggle - Cornell University arXiv Dataset](https://www.kaggle.com/datasets/Cornell-University/arxiv)

---

## 🧪 Compute Similarity Matrix

Different scripts are provided depending on the type of data you want to cluster:

### 📄 Paper-to-Paper Similarity
Run `compute_paper_to_paper_similarity.py` with the following arguments:

```bash
--embedding     # Embedding type (default: 'TF-IDF')
--num_samples   # Number of samples to use (default: 200)
--metric        # Clustering metric (default: 'cosine')
--PCA           # Apply PCA (flag)
--g             # Plot PCA explained variance (flag)
--MC            # Max number of categories (default: 6)
--balance       # Category balance factor (default: 0.1)
```

### 📚 Subcategory-to-Subcategory Similarity
Run `compute_subcat_to_subcat_similarity.py` with:

```bash
--embedding     # Embedding type (default: 'TF-IDF')
--num_samples   # Number of samples to use (default: 1000)
--metric        # Clustering metric (default: 'cosine')
--PCA           # Apply PCA (flag)
--g             # Plot PCA explained variance (flag)
--MC            # Max number of categories
--balance       # Category balance factor (default: 0.1)
```

### 🌐 Wikipedia Similarity (Abstracts and Adjacency)
Run `compute_wikipedia_similarity.py`:

```bash
--embedding     # Embedding type (default: 'TF-IDF')
--num_samples   # Number of samples to use (default: 200)
--metric        # Clustering metric (default: 'cosine')
--PCA           # Apply PCA (for the abstract matrix) (flag)
--g             # Plot PCA explained variance (flag)
```

---

## 🧩 Run Hierarchical Clustering

Use `main.py` after computing the similarity matrix:

```bash
--embedding     # Embedding type (default: 'TF-IDF')
--num_samples   # Number of samples to use (default: 100)
--metric        # Linkage metric (same as for similarity)
--eps           # Epsilon value for Algorithm 3
--delta         # Delta value for Algorithm 3 (default: 0.1)
--deltaType     # Type of delta algorithm (default: 2)
--PCA           # Apply PCA (flag)
--dataset       # Dataset: 'wikipedia', 'categories', or 'abstracts'
--lead          # if the flag is use it will cluster on abstract else it will cluster on the adjency matrix
```

---

## 📁 Project Structure

```
hierarchical_clustering/
│
├── resources/                        # Dataset location
│   └── arxiv-metadata-oai-snapshot.json
├── compute_paper_to_paper_similarity.py
├── compute_subcat_to_subcat_similarity.py
├── compute_wikipedia_similarity.py
├── main.py
├── README.md
```

---

## 📌 Notes

-- To enable GPU compatibility, modify main.py as mention in the comments

---

## 📜 License

This project is licensed under the MIT License.
