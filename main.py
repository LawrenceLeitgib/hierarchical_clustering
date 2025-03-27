import scipy as sp
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.metrics import pairwise_distances

from matplotlib import pyplot as plt
import json
import argparse
import cupy as cp


# Limit managed memory usage to 32GB
MAX_RAM_USAGE = 32 * (1024**3)/1000  # 32GB in bytes
cp.cuda.MemoryPool(cp.cuda.malloc_managed).set_limit(MAX_RAM_USAGE)
cp.cuda.set_allocator(cp.cuda.MemoryPool(cp.cuda.malloc_managed).malloc)

from src.plottingTree import plot_tree
from src.treeSet import Node,tree_set_to_tree
from scipy.spatial.distance import squareform


def main(embedding, num_samples, metric,eps,delta,deltaType,PCA,c):
    print(f"Using embedding: {embedding}")


   
    
    PCA_Flag = "PCA_" if PCA else ""

    c_flag = "c_" if c else ""

    distance_matrix = cp.load("distance_matrix/distance_matrix_"+c_flag+embedding+"_"+PCA_Flag+str(num_samples)+".npy")
    n=num_samples
    if(c):
        ##load the categories list
        import numpy as np
        categories_list = np.load("categories_list/categories_list_"+str(num_samples)+".npy",allow_pickle=True)
        categories = sorted(set(categories_list))
        n= len(categories)
    similarity_matrix = 1 / (1 + distance_matrix)  # Simple similarity measure
    condensed_distance = squareform(distance_matrix.get(), checks=False)
    print(condensed_distance.shape)
    print(distance_matrix)

    #Alogrith 1: Compute the binary tree
    T_linkage=linkage(condensed_distance,method='complete',metric=metric)
    fig = plt.figure(figsize=(35, 15))
    dn = dendrogram(T_linkage)
    plt.savefig('TF_IDF_T_Linkage.png') 


    #Algorithm 1.5: transform the linkage matrix into a set of sets
    labels = [f"{i}" for i in range(n)]
    set_of_sets = linkage_to_set_of_sets(T_linkage, labels)

    #Algorithm 2: Trim invalid clusters
    if(deltaType==1):
        most_informative_hierarchy = trim_invalid_clusters_1(set_of_sets,labels, similarity_matrix,eps,delta)
    else:
        most_informative_hierarchy = trim_invalid_clusters_2(set_of_sets, similarity_matrix,eps,delta)

    print(set_of_sets)
    print("-----------------------------")
    print(most_informative_hierarchy)

   
    plot_tree(most_informative_hierarchy,num_samples,n)






def trim_invalid_clusters_1(tree, labels,similarity_matrix, eps,delta):
    """
    Trims clusters that do not satisfy the similarity condition.
    """
    valid_clusters = []
    num_samples = similarity_matrix.shape[0]

    for cluster in tree:

        # If cluster size is 1, automatically valid
        if len(cluster) < 2:
            valid_clusters.append(cluster)
            continue

        if(is_valide_cluster_rule(cluster,set(labels)-set(cluster) ,similarity_matrix, eps, delta)):
            valid_clusters.append(cluster)
    return valid_clusters

def trim_invalid_clusters_2(tree, similarity_matrix, eps,delta):
    tree_node=tree_set_to_tree(tree)

    valid_clusters = [tree_node.data]

    for c in tree_node.children:
       trim_invalid_clusters_2_rec(c, similarity_matrix, eps,delta,valid_clusters)

    valid_clusters.sort(key=lambda x: len(x),reverse=False)


    return valid_clusters

def trim_invalid_clusters_2_rec(tree_node, similarity_matrix, eps,delta,valid_clusters):
   
    if is_valide_cluster_parent(tree_node,tree_node.parent, similarity_matrix, eps, delta):
        valid_clusters.append(tree_node.data)
        for c in tree_node.children:
            trim_invalid_clusters_2_rec(c, similarity_matrix, eps,delta,valid_clusters)
    else:
        for c in tree_node.children:
            c.parent = tree_node.parent
        for c in tree_node.children:
            trim_invalid_clusters_2_rec(c, similarity_matrix, eps,delta,valid_clusters)


def is_valide_cluster_parent(c,p, similarity_matrix, eps, delta):
    if(len(c.children)==0):
        assert len(c.data)==1
        return True
    parentWithoutCluster=set(p.data)-set(c.data)
    return(is_valide_cluster_rule(c.data,parentWithoutCluster, similarity_matrix, eps, delta))

    
def is_valide_cluster_rule(cluster,nonCluster, similarity_matrix, eps, delta):
    if len(nonCluster) == 0:
        return True
    cluster_items = cp.array(list(map(int, cluster)))  # Convert to NumPy array for fast indexing
    nonCluster_items=cp.array(list(map(int, nonCluster)))


    x_idx, y_idx = cp.triu_indices(len(cluster_items), k=1)  # These are relative indices

    x, y = cluster_items[x_idx], cluster_items[y_idx]  # Convert to actual elements
    s_xy = similarity_matrix[x, y]
    s_xz = similarity_matrix[x[:, None], nonCluster_items]  # Shape: (num_pairs, non_cluster_size)
    s_yz = similarity_matrix[y[:, None], nonCluster_items]  # Shape: (num_pairs, non_cluster_size)


    # Compute max(s_xz, s_yz)
    max_s_xz_yz = cp.maximum(s_xz, s_yz)  # Shape: (num_pairs, non_cluster_size)



    # Ensure `s_xy` is reshaped correctly for broadcasting
    s_xy = s_xy[:, None]  # Shape: (num_pairs, 1)

    violation_counts = (max_s_xz_yz - s_xy > -eps).sum(axis=(0, 1))   


    if violation_counts  <= delta*(len(cluster_items)*(len(cluster_items)-1)*len(nonCluster_items)/2):
        return True
    else :
        if(delta>=1):
            print("error-------------------------------")
            quit()
    return False



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
    parser.add_argument('--deltaType', type=int, default=2, help='The algo use for the detla value')
    parser.add_argument('--PCA', type=bool, default=False, help='Apply PCA')
    parser.add_argument('--c', type=bool, default=False, help='use categories')

    args = parser.parse_args()
    main(args.embedding, args.num_samples, args.metric,args.eps,args.delta,args.deltaType,args.PCA,args.c)