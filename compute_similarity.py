from sklearn.feature_extraction.text import TfidfVectorizer,CountVectorizer
from sklearn.metrics import pairwise_distances

from matplotlib import pyplot as plt
import json
import argparse
import cupy as np



NUMBER_OF_SAMPLE=50


def main(embedding, num_samples, metric,PCA,mc,balance):
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
        matrix=apply_PCA(matrix)

    print(matrix.shape)
    
    distance_matrix = pairwise_distances(matrix, metric=metric)
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
    plt.savefig("categories_chart/"+str(num_samples)+"_.png")

    PCA_Flag = "PCA_" if PCA else ""


    np.save("distance_matrix/distance_matrix_"+embedding+"_"+PCA_Flag+str(num_samples)+".npy", distance_matrix)
    np.save("categories_list/categories_map_"+str(num_samples)+".npy", categories_list)


    #creat the PCA plot
    if(PCA):
        plt.figure(figsize=(10, 6))
        plt.scatter(matrix[:, 0], matrix[:, 1])
        plt.title('PCA Plot')
        plt.savefig("PCA_plot/"+str(num_samples)+"_.png")


    

def extract_abstracts_and_categories(num_samples,mc,balance):
    with open('resources/arxiv-metadata-oai-snapshot.json') as f:
        abstract = []
        categories_dist ={}
        categories_list=[]
        count=0
        for line in f:
            c=json.loads(line)['categories']
            #contiune if c contain a space
            if ' ' in c:
                continue

            if(len(categories_dist.keys())==mc and c not in categories_dist):
                continue

            if(categories_dist.get(c,0)>=((1+balance)*(num_samples/mc)) and balance!=-1):
                continue

            abstract.append(json.loads(line)['abstract'])
            categories_list.append(json.loads(line)['categories'])
            if c not in categories_dist:
                categories_dist[c]=1
            else:
                categories_dist[c]+=1

            count+=1
            if count==num_samples:
                break
  
    return abstract,categories_dist,categories_list

def apply_SVD(matrix):
    from sklearn.decomposition import TruncatedSVD
    import numpy as np
    
    # Determine an upper bound for components: usually min(n_samples, n_features)
    n_components_max = min(matrix.shape)
    
    # First, compute SVD with maximum components to get the explained variance ratios
    svd_full = TruncatedSVD(n_components=n_components_max)
    svd_full.fit(matrix)
    cumulative_variance = np.cumsum(svd_full.explained_variance_ratio_)
    
    # Find the smallest number of components that explain at least 90% of the variance
    n_components_required = np.searchsorted(cumulative_variance, 0) + 2
    
    # Now run TruncatedSVD with the selected number of components
    svd = TruncatedSVD(n_components=n_components_required)
    principalComponents = svd.fit_transform(matrix)
    
    print(f"Number of components selected: {n_components_required}")
    return principalComponents
def apply_PCA(matrix):
    from sklearn.decomposition import PCA,TruncatedSVD
    import numpy as np
       # Determine an upper bound for components: usually min(n_samples, n_features)
    n_components_max = min(matrix.shape)
    print(matrix.shape)
    
    # First, compute SVD with maximum components to get the explained variance ratios
    svd_full = TruncatedSVD(n_components=n_components_max)
    svd_full.fit(matrix)
    cumulative_variance = np.cumsum(svd_full.explained_variance_ratio_)
    
    # Find the smallest number of components that explain at least 90% of the variance
    n_components_required = np.searchsorted(cumulative_variance, 0.1) + 1
    pca = PCA(n_components=n_components_required)
    principalComponents = pca.fit_transform(matrix)
    return principalComponents


def bm25_transform(corpus, k1=1.5, b=0.75):
    vectorizer = CountVectorizer()
    term_freq_matrix = vectorizer.fit_transform(corpus)  # Get raw term frequencies
    doc_lengths = np.array(term_freq_matrix.sum(axis=1)).flatten()  # Convert to NumPy array
    avg_doc_length = np.mean(doc_lengths)  # Average document length

    # Compute IDF values
    df = np.array((term_freq_matrix > 0).sum(axis=0)).flatten()  # Convert sparse to dense
    total_docs = len(corpus)
    idf = np.log((total_docs - df + 0.5) / (df + 0.5) + 1)  # BM25 IDF formula

    # Compute BM25 scores
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
    parser.add_argument('--PCA', type=bool, default=False, help='Apply PCA')
    parser.add_argument('--MC', type=int, default=1000, help='max number of categories')
    parser.add_argument('--balance', type=float, default=-1, help='use to balance the number of categories')
    args = parser.parse_args()
    main(args.embedding, args.num_samples, args.metric,args.PCA,args.MC,args.balance)