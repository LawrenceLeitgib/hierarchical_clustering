from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import pairwise_distances
from collections import defaultdict
from compute_similarity import bm25_transform,apply_PCA,extract_abstracts_and_categories

from IPython.display import SVG

import numpy as np
from scipy.cluster.hierarchy import linkage


from sknetwork.data import load_netset
from sknetwork.ranking import PageRank, top_k
from sknetwork.embedding import Spectral
from sknetwork.clustering import Louvain
from sknetwork.classification import DiffusionClassifier
from sknetwork.utils import get_neighbors
from sknetwork.visualization import visualize_dendrogram





from matplotlib import pyplot as plt
import matplotlib.image as mpimg
from io import BytesIO
import cairosvg

import argparse
import numpy as np



NUMBER_OF_SAMPLE=10000

def main(args):
    wikivitals = load_netset('wikivitals')
    adjacency = wikivitals.adjacency
    labels = wikivitals.labels
    names = wikivitals.names


    #print all attributes of wikivitals
    

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
    # hierarchy of articles
    label = label_id['Physical sciences']
    index = selection[label]
    dendrogram_articles = linkage(embedding[(index)], method='ward')


    new_names = []
    for i in index:
        new_names.append(names[i]+" ("+names_labels[labels[i]]+")")
    print(new_names)
    np.save(f"wikipedia_labels/wikivitals_names_{args.num_samples}.npy", new_names)
    #print(dendrogram_articles)
    image = visualize_dendrogram(dendrogram_articles, names=new_names, rotate=True, width=200, scale=2, n_clusters=4)

    distance_matrix = pairwise_distances(embedding[index], metric=args.metric)
    # Convert SVG string to PNG
    png_data = cairosvg.svg2png(bytestring=image.encode('utf-8'))
    img = mpimg.imread(BytesIO(png_data), format='png')

    # Show image
    plt.imshow(img)
    plt.axis('off')
    print(wikivitals.keys())
   # plt.show()

    np.save(f"distance_matrix/distance_matrix_w_{args.num_samples}.npy", distance_matrix)
  



    #print(dendrogram_articles)












    

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Specify embedding type.')
    parser.add_argument('--num_samples', type=int, default=NUMBER_OF_SAMPLE, help='Number of samples to use')
    parser.add_argument('--metric', type=str, default='cosine', help='The metric to use for clustering')
    parser.add_argument('--PCA', type=bool, default=False, help='Apply PCA')
    parser.add_argument('--g', type=bool, default=False, help='wether to use plot PCA explained variance')
    args = parser.parse_args()
    main(args)