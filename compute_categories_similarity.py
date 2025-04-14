from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import pairwise_distances
from collections import defaultdict
from compute_similarity import bm25_transform,apply_PCA,extract_abstracts_and_categories
import json


from matplotlib import pyplot as plt
import argparse
import numpy as np

#count the number of categories in the .json file
n_categories = 0
with open('resources/categories_name_map.json') as json_file:
    n_categories = len(json.load(json_file))
print(f"Number of categories: {n_categories}")


NUMBER_OF_SAMPLE=10000
NUMBER_OF_SUB_CAT=n_categories

def main(embedding, num_samples, metric,PCA,mc,balance,g):
    print(f"Using embedding: {embedding}")
    abstract,categories_dict,categories_list = extract_abstracts_and_categories(num_samples,mc,balance)

    #Embedding the abstracts
    if embedding == 'TF-IDF':
        vectorizer = TfidfVectorizer()
        matrix = vectorizer.fit_transform(abstract)
        
    elif embedding == 'BM25':
        matrix,vectorizer=bm25_transform(abstract)
    else:
        raise ValueError("Invalid embedding type")


    if(PCA):
        matrix=apply_PCA(matrix,g)

    print(matrix.shape)
    
    distance_matrix = pairwise_distances(matrix, metric=metric)


        ##use the matrix and the categories to averge the similarity between the categories and creat a categories_simialrity matrix
    categories = sorted(set(categories_list))
    k = len(categories)

   # Map from category to list of sample indices
    cat_to_samples = defaultdict(list)
    for idx, cat in enumerate(categories_list):
        cat_to_samples[cat].append(idx)

    category_distances = np.zeros((k, k))

    # Fill upper triangle (and mirror for symmetry)
    for i, cat_i in enumerate(categories):
        for j in range(i, k):
            cat_j = categories[j]
            samples_i = cat_to_samples[cat_i]
            samples_j = cat_to_samples[cat_j]
            # Extract all distances between samples of cat_i and cat_j
            dists = distance_matrix[np.ix_(samples_i, samples_j)]
            mean_dist = np.mean(dists)
            category_distances[i, j] = mean_dist
            category_distances[j, i] = mean_dist


    #urn into a DataFrame for readability
    import pandas as pd
    category_similarity_df = pd.DataFrame(category_distances, index=categories, columns=categories)
    print("Category similarity matrix:")
    print(category_similarity_df)
        
    # Extracting labels and values
    new_categories = {}
    for key, value in categories_dict.items():
        if(value >= num_samples*0.01):
            new_categories[key] = value
        else:
            new_categories['Other'] = new_categories.get('Other', 0) + value

    labels = list(new_categories.keys())
    values = list(new_categories.values())

    # Creating the pie chart
    plt.figure(figsize=(10, 6))
    plt.pie(values, labels=labels, autopct='%1.1f%%', startangle=140, wedgeprops={'edgecolor': 'black'})

    # Adding a title
    plt.title('Pie Chart of Categories')
    plt.savefig("out/categories_chart/"+str(num_samples)+"_.png")

    PCA_Flag = "PCA_" if PCA else ""


    np.save("distance_matrix/distance_matrix_c_"+embedding+"_"+PCA_Flag+str(num_samples)+".npy", category_distances)
    np.save("categories_list/categories_list_"+str(num_samples)+".npy", categories)


    #creat the PCA plot
    if(PCA):
        plt.figure(figsize=(10, 6))
        plt.scatter(matrix[:, 0], matrix[:, 1])
        plt.title('PCA Plot')
        plt.savefig("out/PCA_plot/"+str(num_samples)+"_.png")


    

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Specify embedding type.')
    parser.add_argument('--embedding', type=str, default='TF-IDF', help='The type of embedding to use')
    parser.add_argument('--num_samples', type=int, default=NUMBER_OF_SAMPLE, help='Number of samples to use')
    parser.add_argument('--metric', type=str, default='cosine', help='The metric to use for clustering')
    parser.add_argument('--PCA', type=bool, default=False, help='Apply PCA')
    parser.add_argument('--g', type=bool, default=False, help='wether to use plot PCA explained variance')

    parser.add_argument('--MC', type=int, default=NUMBER_OF_SUB_CAT, help='max number of categories')
    parser.add_argument('--balance', type=float, default=-1, help='use to balance the number of categories')
    args = parser.parse_args()
    main(args.embedding, args.num_samples, args.metric,args.PCA,args.MC,args.balance,args.g)