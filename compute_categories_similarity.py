from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import pairwise_distances
from collections import defaultdict
from compute_similarity import bm25_transform, apply_PCA, extract_abstracts_and_categories
import json
from matplotlib import pyplot as plt
import argparse
import numpy as np
import pandas as pd

# Count the number of categories in the .json file
with open('resources/categories_name_map.json') as json_file:
    n_categories = len(json.load(json_file))
print(f"Number of categories: {n_categories}")

NUMBER_OF_SAMPLE = 10000
NUMBER_OF_SUB_CAT = n_categories


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
        matrix = apply_PCA(matrix, args.g)

    print(matrix.shape)
    distance_matrix = pairwise_distances(matrix, metric=args.metric)

    # Compute category-wise average similarities
    categories = sorted(set(categories_list))
    k = len(categories)
    cat_to_samples = defaultdict(list)
    for idx, cat in enumerate(categories_list):
        cat_to_samples[cat].append(idx)

    category_distances = np.zeros((k, k))
    for i, cat_i in enumerate(categories):
        for j in range(i, k):
            samples_i = cat_to_samples[cat_i]
            samples_j = cat_to_samples[categories[j]]
            dists = distance_matrix[np.ix_(samples_i, samples_j)]
            mean_dist = np.mean(dists)
            category_distances[i, j] = mean_dist
            category_distances[j, i] = mean_dist

    category_similarity_df = pd.DataFrame(category_distances, index=categories, columns=categories)
    print("Category similarity matrix:")
    print(category_similarity_df)

    # Pie chart of categories
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
    np.save(f"distance_matrix/distance_matrix_c_{args.embedding}_{PCA_Flag}{args.num_samples}.npy", category_distances)
    np.save(f"categories_list/categories_list_{args.num_samples}.npy", categories)

    if args.PCA:
        plt.figure(figsize=(10, 6))
        plt.scatter(matrix[:, 0], matrix[:, 1])
        plt.title('PCA Plot')
        plt.savefig(f"out/PCA_plot/{args.num_samples}_.png")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Specify embedding type.')
    parser.add_argument('--embedding', type=str, default='TF-IDF', help='The type of embedding to use')
    parser.add_argument('--num_samples', type=int, default=NUMBER_OF_SAMPLE, help='Number of samples to use')
    parser.add_argument('--metric', type=str, default='cosine', help='The metric to use for clustering')
    parser.add_argument('--PCA', type=bool, default=False, help='Apply PCA')
    parser.add_argument('--g', type=bool, default=False, help='Whether to plot PCA explained variance')
    parser.add_argument('--MC', type=int, default=NUMBER_OF_SUB_CAT, help='Max number of categories')
    parser.add_argument('--balance', type=float, default=-1, help='Balance the number of categories')
    
    args = parser.parse_args()
    main(args)
