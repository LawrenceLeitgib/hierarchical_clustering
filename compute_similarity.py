from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import pairwise_distances

from matplotlib import pyplot as plt
import json
import argparse
import cupy as np



NUMBER_OF_SAMPLE=50


def main(embedding, num_samples, metric,eps):
    print(f"Using embedding: {embedding}")
    abstract = extract_abstracts(num_samples)

    #Embedding the abstracts
    if embedding == 'TF-IDF':
        vectorizer = TfidfVectorizer()
        tfidf_matrix = vectorizer.fit_transform(abstract)
        
    elif embedding == 'BM25':
        print("BM25 not implemented")
    else:
        raise ValueError("Invalid embedding type")
    
    distance_matrix = pairwise_distances(tfidf_matrix, metric=metric)

    np.save("distance_matrix/distance_matrix_"+embedding+"_"+str(num_samples)+".npy", distance_matrix)
    

def extract_abstracts(num_samples):
    with open('resources/arxiv-metadata-oai-snapshot.json') as f:
        abstract = []
        count=0
        for line in f:
            abstract.append(json.loads(line)['abstract'])

            if(count==46 or count==49):
                print(abstract[count])
                pass

            count+=1
            if count==num_samples:
                break
  
    return abstract 





if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Specify embedding type.')
    parser.add_argument('--embedding', type=str, default='TF-IDF', help='The type of embedding to use')
    parser.add_argument('--num_samples', type=int, default=NUMBER_OF_SAMPLE, help='Number of samples to use')
    parser.add_argument('--metric', type=str, default='cosine', help='The metric to use for clustering')
    parser.add_argument('--eps', type=float, default=0, help='The epsilon value for Algorithm 3')
    args = parser.parse_args()
    main(args.embedding, args.num_samples, args.metric,args.eps)