from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.metrics import pairwise_distances
from sklearn.decomposition import PCA, TruncatedSVD

from matplotlib import pyplot as plt
import json
import argparse
import numpy as np

NUMBER_OF_SAMPLE = 100

def main(args):
    print(f"Using embedding: {args.embedding}")

    abstract, categories_dict, categories_list = extract_abstracts_and_categories(
        args.num_samples, args.MC, args.balance
    )

    # Embedding the abstracts
    if args.embedding == 'TF-IDF':
        vectorizer = TfidfVectorizer()
        matrix = vectorizer.fit_transform(abstract)
    elif args.embedding == 'BM25':
        matrix, vectorizer = bm25_transform(abstract)
    elif args.embedding == 'Word2Vec':
        from gensim.models import Word2Vec
        tokenized_abstracts = [doc.split() for doc in abstract]
        model = Word2Vec(tokenized_abstracts, vector_size=100, window=5, min_count=1, workers=4)
        matrix = np.array([
            np.mean([model.wv[word] for word in doc if word in model.wv], axis=0)
            for doc in tokenized_abstracts
        ])
    else:
        raise ValueError("Invalid embedding type")

    if args.PCA:
        matrix = apply_PCA(matrix, 11,args.g)

    print(matrix.shape)

    distance_matrix = pairwise_distances(matrix, metric=args.metric)

    # Extracting labels and values
    new_categories = {}
    for key, value in categories_dict.items():
        if value >= args.num_samples * 0.01:
            new_categories[key] = value
        else:
            new_categories['Other'] = new_categories.get('Other', 0) + value

    labels = list(new_categories.keys())
    values = list(new_categories.values())

    plt.figure(figsize=(10, 6))
    plt.pie(values, labels=labels, autopct='%1.1f%%', startangle=140, wedgeprops={'edgecolor': 'black'})
    plt.title('Pie Chart of Categories')
    plt.savefig(f"out/categories_chart/{args.num_samples}_.png")

    PCA_Flag = "PCA_" if args.PCA else ""
    np.save(f"distance_matrix/distance_matrix_{args.embedding}_{PCA_Flag}{args.num_samples}.npy", distance_matrix)
    np.save(f"categories_list/categories_map_{args.num_samples}.npy", categories_list)
    true_categories = get_true_categories(categories_list)
    np.save(f"true_categories/true_categories_abstracts_{args.num_samples}.npy",np.array(true_categories, dtype=object))

    if args.PCA:
        plt.figure(figsize=(10, 6))
        plt.scatter(matrix[:, 0], matrix[:, 1])
        plt.title('PCA Plot')
        plt.savefig(f"out/PCA_plot/{args.num_samples}_.png")

def get_true_categories(categories_list):
    """
    Create a set of sets of true categories from the categories list.
    """
    categories = sorted(set(categories_list))
    k = len(categories)
    true_categories_dict={}
    for i in range(k):
        true_categories_dict[categories[i]] = set()
    for i in range(len(categories_list)):
        true_categories_dict[categories_list[i]].add(str(i))
    true_categories = list()
    for i in range(k):
        true_categories.append(tuple(true_categories_dict[categories[i]]))
    return true_categories
   
def extract_abstracts_and_categories(num_samples, mc, balance):
    with open('resources/arxiv-metadata-oai-snapshot.json') as f:
        abstract = []
        categories_dist = {}
        categories_list = []
        collected_count = 0

        for line in f:
            c = json.loads(line)['categories']
            if ' ' in c:
                continue
            if len(categories_dist) == mc and c not in categories_dist:
                continue
            if categories_dist.get(c, 0) >= ((1 + balance) * (num_samples / mc)) and balance != -1:
                continue

            abstract.append(json.loads(line)['abstract'])
            categories_list.append(c)
            categories_dist[c] = categories_dist.get(c, 0) + 1

            collected_count += 1
            if collected_count % 100 == 0:
                print("number of samples collected:", collected_count, "/", num_samples)
            if collected_count == num_samples:
                break

    return abstract, categories_dist, categories_list


def apply_SVD(matrix):
    n_components_max = min(matrix.shape)
    svd_full = TruncatedSVD(n_components=n_components_max)
    svd_full.fit(matrix)
    cumulative_variance = np.cumsum(svd_full.explained_variance_ratio_)
    n_components_required = np.searchsorted(cumulative_variance, 0) + 2
    svd = TruncatedSVD(n_components=n_components_required)
    principalComponents = svd.fit_transform(matrix)
    print(f"Number of components selected: {n_components_required}")
    return principalComponents


def apply_PCA(matrix,nc, g):
    if g:
        pca = PCA()
        pca.fit(matrix)
        eigenvalues = pca.explained_variance_[:100]
        plt.figure(figsize=(8, 5))
        plt.plot(np.arange(1, len(eigenvalues) + 1), eigenvalues, 'o-', linewidth=2, markersize=6)
        plt.title('Eigenvalues from PCA')
        plt.xlabel('Principal Component')
        plt.ylabel('Eigenvalue')
        plt.grid(True, which="both", ls="--", linewidth=0.5)
        plt.xticks(np.arange(1, len(eigenvalues) + 1))
        plt.savefig(f"out/PCA_eigenValue/{matrix.shape[0]}_.png")

    pca = PCA(n_components=nc)
    return pca.fit_transform(matrix)


def bm25_transform(corpus, k1=1.5, b=0.75):
    vectorizer = CountVectorizer()
    term_freq_matrix = vectorizer.fit_transform(corpus)
    doc_lengths = np.array(term_freq_matrix.sum(axis=1)).flatten()
    avg_doc_length = np.mean(doc_lengths)
    df = np.array((term_freq_matrix > 0).sum(axis=0)).flatten()
    total_docs = len(corpus)
    idf = np.log((total_docs - df + 0.5) / (df + 0.5) + 1)

    bm25_matrix = []
    for i in range(total_docs):
        row = term_freq_matrix[i].toarray().flatten()
        doc_len = doc_lengths[i]
        bm25_row = idf * ((row * (k1 + 1)) / (row + k1 * (1 - b + b * doc_len / avg_doc_length)))
        bm25_matrix.append(bm25_row)

    return np.array(bm25_matrix), vectorizer


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Specify embedding type.')
    parser.add_argument('--embedding', type=str, default='TF-IDF', help='The type of embedding to use')
    parser.add_argument('--num_samples', type=int, default=NUMBER_OF_SAMPLE, help='Number of samples to use')
    parser.add_argument('--metric', type=str, default='cosine', help='The metric to use for clustering')
    parser.add_argument('--PCA', action="store_true", default=False, help='Apply PCA')
    parser.add_argument('--g', action="store_true", default=False, help='Whether to plot PCA explained variance')
    parser.add_argument('--MC', type=int, default=6, help='Max number of categories')
    parser.add_argument('--balance', type=float, default=0.1, help='Balance the number of categories')

    args = parser.parse_args()
    main(args)
