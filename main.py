import scipy as sp
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.metrics import pairwise_distances

from matplotlib import pyplot as plt
import json
import argparse
import numpy as np

from src.plottingTree import plot_tree
from scipy.spatial.distance import squareform


def main(embedding, num_samples, metric,eps,delta):
    print(f"Using embedding: {embedding}")
   

    distance_matrix = np.load("distance_matrix_"+embedding+"_"+str(num_samples)+".npy")
    similarity_matrix = 1 / (1 + distance_matrix)  # Simple similarity measure
    condensed_distance = squareform(distance_matrix, checks=False)
    print(condensed_distance.shape)

    #Alogrith 1: Compute the binary tree
    T_linkage=linkage(condensed_distance,method='single',metric=metric)
    print(T_linkage)
    print("££££££££££££££££££££££££££££££££££$")
    fig = plt.figure(figsize=(35, 15))
    dn = dendrogram(T_linkage)
    plt.savefig('TF_IDF_T_Linkage.png') 


    #Algorithm 1.5: transform the linkage matrix into a set of sets
    labels = [f"{i}" for i in range(num_samples)]
    set_of_sets = linkage_to_set_of_sets(T_linkage, labels)

    #Algorithm 2: Trim invalid clusters
    most_informative_hierarchy = trim_invalid_clusters_1(set_of_sets, similarity_matrix,eps,delta)

    print(set_of_sets)
    print("-----------------------------")
    print(most_informative_hierarchy)

    s_test=[('0',), ('1',), ('2',), ('3',), ('4',), ('5',), ('6',), ('7',),  ('8',), ('9',), ('10',),('2', '6'), ('0', '5'), ('7', '1', '4', '2', '6', '0', '5'),('9','8') ,('3', '7', '1', '4', '2', '6', '0', '5','8','9','10')]
    s_test2=[('0',), ('1',), ('2',), ('3',), ('4',), ('5',), ('6',), ('7',),  ('8',), ('9',), ('10',),('11',),('12',),
             ('0','1'), ('2','3'), ('4','5'), ('6','7'), ('8','9'), ('10','11'),('6','7', '8','9','10','11'),('0','1','2','3','4','5'),('0','1','2','3','4','5','6','7','8','9','10','11','12')]
    plot_tree(s_test2)






def trim_invalid_clusters_1(tree, similarity_matrix, eps,delta):
    """
    Trims clusters that do not satisfy the similarity condition.
    """
    valid_clusters = []
    num_samples = similarity_matrix.shape[0]


    c=0
    for cluster in tree:
        #print(c,len(tree))
        c+=1

        cluster_items = np.array(list(map(int, cluster)))  # Convert to NumPy array for fast indexing

        # If cluster size is 1, automatically valid
        if len(cluster_items) < 2:
            valid_clusters.append(cluster)
            continue

        # Get similarity values for all pairs in the cluster
        # Get all unique pairs (i, j) within the cluster
        x_idx, y_idx = np.triu_indices(len(cluster_items), k=1)  # These are relative indices
        x, y = cluster_items[x_idx], cluster_items[y_idx]  # Convert to actual elements
        s_xy = similarity_matrix[x, y]

        # Get similarity values for non-cluster elements
        non_cluster_items = np.setdiff1d(np.arange(num_samples), cluster_items, assume_unique=True)

      
        # If no non-cluster items exist, no violations can occur
        if len(non_cluster_items) == 0:
            valid_clusters.append(cluster)
            continue

        # Compute similarity values for all (x, z) and (y, z) pairs
        s_xz = similarity_matrix[x[:, None], non_cluster_items]  # Shape: (num_pairs, non_cluster_size)
        s_yz = similarity_matrix[y[:, None], non_cluster_items]  # Shape: (num_pairs, non_cluster_size)

        # Compute max(s_xz, s_yz)
        max_s_xz_yz = np.maximum(s_xz[:, None, :], s_yz[None, :, :])  # Shape: (cluster_size, cluster_size, non_cluster_size)


        # Ensure `s_xy` is reshaped correctly for broadcasting
        s_xy = s_xy[:, None]  # Shape: (num_pairs, 1)

        # Check violation condition: max(s_xz, s_yz) - s_xy > eps
        violation_counts = (max_s_xz_yz - s_xy > eps).sum()
        '''
        print(x,y)
        print(s_xy)
        print(cluster_items)
        print(non_cluster_items)
        print(s_xz,s_yz)
        print(violation_counts)
        print("--------------------------------------------------")
        '''

        # If fewer than 2 violations, keep the cluster
        if (violation_counts  < delta*num_samples):
            valid_clusters.append(cluster)


    return valid_clusters




def linkage_to_set_of_sets(T_linkage, labels):
    n = len(labels)
    clusters = {i: (labels[i],) for i in range(n)}  


    for i, (c1, c2, _, _) in enumerate(T_linkage):
        c1, c2 = int(c1), int(c2)

        # Merge clusters
        new_cluster = clusters[c1] + clusters[c2]
        clusters[n + i] = new_cluster

        # Store cluster as a set or tuple
    return [value for value in clusters.values()]




if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Specify embedding type.')
    parser.add_argument('--embedding', type=str, default='TF-IDF', help='The type of embedding to use')
    parser.add_argument('--num_samples', type=int, default=50, help='Number of samples to use')
    parser.add_argument('--metric', type=str, default='cosine', help='The metric to use for clustering')
    parser.add_argument('--eps', type=float, default=0, help='The epsilon value for Algorithm 3')
    parser.add_argument('--delta', type=float, default=0.1, help='The delta value for Algorithm 3')
    args = parser.parse_args()
    main(args.embedding, args.num_samples, args.metric,args.eps,args.delta)