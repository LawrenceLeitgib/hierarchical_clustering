import scipy as sp
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.cluster.hierarchy import linkage

from sknetwork.data import load_netset
from sknetwork.ranking import PageRank, top_k
from sknetwork.embedding import Spectral
from sknetwork.clustering import Louvain
from sknetwork.classification import DiffusionClassifier
from sknetwork.utils import get_neighbors
from sknetwork.visualization import visualize_dendrogram





from matplotlib import pyplot as plt
import json
import argparse
import cupy as cp

# Limit managed memory usage to 32GB
MAX_RAM_USAGE = 32 * (1024**3) / 1000  # 32GB in bytes
cp.cuda.MemoryPool(cp.cuda.malloc_managed).set_limit(MAX_RAM_USAGE)
cp.cuda.set_allocator(cp.cuda.MemoryPool(cp.cuda.malloc_managed).malloc)

from src.plottingTree import plot_tree
from src.treeSet import Node, tree_set_to_tree
from scipy.spatial.distance import squareform




def main(args):
    print(f"Using embedding: {args.embedding}")
    n = args.num_samples


    if(args.dataset=="wikipedia"):
        import numpy as np


        wikivitals = load_netset('wikivitals')
        adjacency = wikivitals.adjacency
        names = wikivitals.names
        labels = wikivitals.labels
        names_labels = wikivitals.names_labels
        label_id = {name: i for i, name in enumerate(names_labels)}
        pagerank = PageRank()
        n_selection = args.num_samples
        # selection of articles
        selection = []
        for label in np.arange(len(names_labels)):
            ppr = pagerank.fit_predict(adjacency, weights=(labels==label))
            scores = ppr * (labels==label)
            selection.append(top_k(scores, n_selection))
        selection = np.array(selection)

        n_components = 20
        # embedding
        spectral = Spectral(n_components)
        embedding = spectral.fit_transform(adjacency)
        print(embedding.shape)
        label = label_id['Physical sciences']
        index = selection[label]
        T_linkage = linkage(embedding[index], method='ward')


    PCA_Flag = "PCA_" if args.PCA else ""
    c_flag = "c_" if args.dataset=="categories" else ""
    if(args.dataset=="wikipedia"):
        distance_matrix = cp.load(f"distance_matrix/distance_matrix_w_{args.num_samples}.npy")
    else:
        distance_matrix = cp.load(f"distance_matrix/distance_matrix_{c_flag}{args.embedding}_{PCA_Flag}{args.num_samples}.npy")

    if args.dataset=="categories":
        import numpy as np
        categories_list = np.load(f"categories_list/categories_list_{args.num_samples}.npy", allow_pickle=True)
        categories = sorted(set(categories_list))
        n = len(categories)

    similarity_matrix = 1 / (1 + distance_matrix)
    condensed_distance = squareform(distance_matrix.get(), checks=False)

    print(condensed_distance.shape)
    print(distance_matrix)

    # Algorithm 1: Compute the binary tree
    T_linkage = linkage(condensed_distance, method='complete', metric=args.metric)

    labels = [f"{i}" for i in range(n)]
    set_of_sets = linkage_to_set_of_sets(T_linkage, labels)

    # Algorithm 2: Trim invalid clusters
    if args.deltaType == 1:
        most_informative_hierarchy = trim_invalid_clusters_1(set_of_sets, labels, similarity_matrix, args.eps, args.delta)
    else:
        most_informative_hierarchy = trim_invalid_clusters_2(set_of_sets, similarity_matrix, args.eps, args.delta)

    print(set_of_sets)
    print("-----------------------------")
    print(most_informative_hierarchy)

    plot_tree(most_informative_hierarchy, args.num_samples, n,args.dataset, isSet=True)
    plot_tree(T_linkage, args.num_samples, n,args.dataset, isSet=False)


def trim_invalid_clusters_1(tree, labels, similarity_matrix, eps, delta):
    valid_clusters = []
    for cluster in tree:
        if len(cluster) < 2 or is_valide_cluster_rule(cluster, set(labels) - set(cluster), similarity_matrix, eps, delta):
            valid_clusters.append(cluster)
    return valid_clusters


def trim_invalid_clusters_2(tree, similarity_matrix, eps, delta):
    tree_node = tree_set_to_tree(tree)
    valid_clusters = [tree_node.data]
    for c in tree_node.children:
        trim_invalid_clusters_2_rec(c, similarity_matrix, eps, delta, valid_clusters)
    valid_clusters.sort(key=lambda x: len(x), reverse=False)
    return valid_clusters


def trim_invalid_clusters_2_rec(tree_node, similarity_matrix, eps, delta, valid_clusters):
    if is_valide_cluster_parent(tree_node, tree_node.parent, similarity_matrix, eps, delta):
        valid_clusters.append(tree_node.data)
        for c in tree_node.children:
            trim_invalid_clusters_2_rec(c, similarity_matrix, eps, delta, valid_clusters)
    else:
        for c in tree_node.children:
            c.parent = tree_node.parent
        for c in tree_node.children:
            trim_invalid_clusters_2_rec(c, similarity_matrix, eps, delta, valid_clusters)


def is_valide_cluster_parent(c, p, similarity_matrix, eps, delta):
    if len(c.children) == 0:
        return True
    parent_without_cluster = set(p.data) - set(c.data)
    return is_valide_cluster_rule(c.data, parent_without_cluster, similarity_matrix, eps, delta)


def is_valide_cluster_rule(cluster, nonCluster, similarity_matrix, eps, delta):
    if len(nonCluster) == 0:
        return True

    cluster_items = cp.array(list(map(int, cluster)))
    nonCluster_items = cp.array(list(map(int, nonCluster)))

    x_idx, y_idx = cp.triu_indices(len(cluster_items), k=1)
    x, y = cluster_items[x_idx], cluster_items[y_idx]
    s_xy = similarity_matrix[x, y]

    s_xz = similarity_matrix[x[:, None], nonCluster_items]
    s_yz = similarity_matrix[y[:, None], nonCluster_items]
    max_s_xz_yz = cp.maximum(s_xz, s_yz)
    s_xy = s_xy[:, None]

    violation_counts = (max_s_xz_yz - s_xy > -eps).sum()

    if violation_counts <= delta * (len(cluster_items) * (len(cluster_items) - 1) * len(nonCluster_items) / 2):
        return True
    else:
        if delta >= 1:
            print("error-------------------------------")
            quit()
    return False


def linkage_to_set_of_sets(T_linkage, labels):
    n = len(labels)
    clusters = {i: (labels[i],) for i in range(n)}
    for i, (c1, c2, _, _) in enumerate(T_linkage):
        c1, c2 = int(c1), int(c2)
        clusters[n + i] = clusters[c1] + clusters[c2]
    return [value for value in clusters.values()]


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Specify embedding type.')
    parser.add_argument('--embedding', type=str, default='TF-IDF', help='The type of embedding to use')
    parser.add_argument('--num_samples', type=int, default=50, help='Number of samples to use')
    parser.add_argument('--metric', type=str, default='cosine', help='The metric to use for clustering')
    parser.add_argument('--eps', type=float, default=0, help='The epsilon value for Algorithm 3')
    parser.add_argument('--delta', type=float, default=0.1, help='The delta value for Algorithm 3')
    parser.add_argument('--deltaType', type=int, default=2, help='The algo use for the delta value')
    parser.add_argument('--PCA', type=bool, default=False, help='Apply PCA')
    parser.add_argument('--dataset', type=str, default="abstract", help='use categories')

    args = parser.parse_args()
    main(args)
