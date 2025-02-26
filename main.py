import scipy as sp
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.metrics import pairwise_distances

from matplotlib import pyplot as plt
import json
import argparse
import numpy as np

from src.plottingTree import build_graph, plot_tree

NUMBER_OF_SAMPLE=8

def main(embedding):
    print(f"Using embedding: {embedding}")
    abstract = extract_abstracts()

    #Embedding the abstracts
    if embedding == 'TF-IDF':
        vectorizer = TfidfVectorizer()
        tfidf_matrix = vectorizer.fit_transform(abstract)
    
    elif embedding == 'BM25':
        print("BM25 not implemented")
    else:
        raise ValueError("Invalid embedding type")
    
    #Alogrith 1: Compute the binary tree
    T_linkage=linkage(tfidf_matrix.toarray(),method='complete',metric='cosine')
    fig = plt.figure(figsize=(35, 15))
    dn = dendrogram(T_linkage)
    plt.savefig('TF_IDF_T_Linkage.png') 


    #Algorithm 1.5: transform the linkage matrix into a set of sets
    labels = [f"{i}" for i in range(len(abstract))]
    set_of_sets = linkage_to_set_of_sets(T_linkage, labels)

    #Algorithm 2: Trim invalid clusters
    distance_matrix = pairwise_distances(tfidf_matrix, metric='cosine')
    similarity_matrix = 1 / (1 + distance_matrix)  # Simple similarity measure
    most_informative_hierarchy = trim_invalid_clusters(set_of_sets, similarity_matrix)

    print(set_of_sets)
    print("-----------------------------")
    print(most_informative_hierarchy)

    G, root = build_graph(most_informative_hierarchy)

    plot_tree(G, root)




def trim_invalid_clusters(tree, similarity_matrix):
    """
    Trims clusters that do not satisfy the similarity condition.
    """
    valid_clusters = []

    for cluster in tree:
        is_valid = True
        cluster_items = list(cluster)
        
        # Check similarity condition for all triplets (x, y, z)
        for i in range(len(cluster_items)):
            for j in range(i + 1, len(cluster_items)):
                x, y = int(cluster_items[i]), int(cluster_items[j])
                s_xy = similarity_matrix[x][y]

                for z in range(similarity_matrix.shape[0]):
                    if str(z) not in cluster_items:
                        s_xz = similarity_matrix[x][z]
                        s_yz = similarity_matrix[y][z]
                        if s_xy < max(s_xz, s_yz):
                            is_valid = False
                            break
                if not is_valid:
                    break

        if is_valid:
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

def extract_abstracts():
    with open('resources/arxiv-metadata-oai-snapshot.json') as f:
        data = []
        count=0
        for line in f:
            data.append(json.loads(line))

            if(count==46 or count==49):
                #print(data[count]['abstract'])
                pass

            count+=1
            if count==NUMBER_OF_SAMPLE:
                break
  
    abstract=[]
    for d in data:
        abstract.append(d['abstract'])
    return abstract # Save the plot to a file








if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Specify embedding type.')
    parser.add_argument('--embedding', type=str, default='TF-IDF', help='The type of embedding to use')
    args = parser.parse_args()
    main(args.embedding)