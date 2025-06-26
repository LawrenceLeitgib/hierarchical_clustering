from typing import Union
from sklearn.metrics import pairwise_distances
import numpy as np

from sknetwork.data import load_netset
from sknetwork.ranking import PageRank, top_k
from sknetwork.embedding import Spectral
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import pairwise_distances
from compute_paper_to_paper_similarity import bm25_transform, apply_PCA, create_all_directories


import json
from pathlib import Path
from typing import Iterable, List, Tuple, Union, Dict

import argparse
import numpy as np





NUMBER_OF_SAMPLE=10000

def main(args):
    create_all_directories()
    wikivitals = load_netset('wikivitals')
    adjacency = wikivitals.adjacency
    labels = wikivitals.labels
    names = wikivitals.names
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
    index=[]
    i=0
    t=0
    set_of_true_categories=[]

    for category in names_labels:
        label = label_id[category]
        c=(n_selection*(i+1))//11-(n_selection*i)//11

        set_of_true_categories.append(tuple(str(i) for i in range(t,c+t)))

        t+=c

        index.append(selection[label][:c])
        i+=1
    index = [item for sublist in index for item in sublist]
    

    assert(t==len(index))

    new_names = []
    for i in index:
        new_names.append((names[i],names_labels[labels[i]]))

    print(set_of_true_categories)

    np.save(f"true_categories/true_categories_wikipedia_{args.num_samples}.npy",np.array(set_of_true_categories, dtype=object))
    np.save(f"wikipedia_labels/wikivitals_names_{args.num_samples}.npy", new_names)
    distance_matrix = pairwise_distances(embedding[index], metric=args.metric)
    np.save(f"distance_matrix/distance_matrix_w_{args.num_samples}.npy", distance_matrix)




    #load the leads
    leads = extractLead(new_names, jsonl_path="resources/vital_abstracts.jsonl")

    for i,l in enumerate(leads):
        if(len(l) <100):
            print(new_names[i],l)

    if args.embedding == 'TF-IDF':
        vectorizer = TfidfVectorizer()
        matrix = vectorizer.fit_transform(leads)
    elif args.embedding == 'BM25':
        matrix, vectorizer = bm25_transform(leads)
    elif args.embedding == 'Word2Vec':
        from gensim.models import Word2Vec
        tokenized_abstracts = [doc.split() for doc in leads]
        model = Word2Vec(tokenized_abstracts, vector_size=100, window=5, min_count=1, workers=4)
        matrix = np.array([
            np.mean([model.wv[word] for word in doc if word in model.wv], axis=0)
            for doc in tokenized_abstracts
        ])
    else:
        raise ValueError("Invalid embedding type") 
    
    if args.PCA:
        matrix = apply_PCA(matrix,11, args.g)

    print(matrix.shape)
    distance_matrix = pairwise_distances(matrix, metric=args.metric)
    PCA_Flag = "PCA_" if args.PCA else ""
    np.save(f"distance_matrix/distance_matrix_wl_{args.num_samples}_{PCA_Flag}{args.embedding}.npy", distance_matrix)


def _load_leads(jsonl_path: Union[str, Path]) -> Dict[str, str]:
    """Read *once* the JSONL file and build a title → lead lookup table."""
    table: Dict[str, str] = {}
    with Path(jsonl_path).expanduser().open("r", encoding="utf‑8") as fh:
        for line in fh:
            rec = json.loads(line)
            table[rec["title"]] = rec["lead"]
    return table


def extractLead(
    article_tuples: Iterable[Tuple[str, ...]],
    *,
    jsonl_path: Union[str, Path] = "resources/vital_abstracts.jsonl",
    default: str | None = None,
) -> List[str]:
    """Return a list of lead texts that matches *article_tuples* order.

    Parameters
    ----------
    article_tuples : Iterable[Tuple[str, ...]]
        An iterable of tuples where **the first element** of each tuple is the
        Wikipedia article title (string). Any extra elements are ignored.

    jsonl_path : str | Path, optional
        Location of the JSONL file produced by *extract_leads.py*. Defaults to
        ``"vital_abstracts.jsonl"``.

    default : str | None, optional
        What to return when a title is *not* found in the JSONL file.
        * ``None``   – raise ``KeyError`` (default).
        * any string – use this as a placeholder.

    Returns
    -------
    list[str]
        Leads in exactly the same order as the incoming tuples.
    """

    lookup = _load_leads(jsonl_path)
    leads: List[str] = []

    for tpl in article_tuples:
        if not tpl:
            raise ValueError("Each tuple must contain at least one element (the title)")
        title = tpl[0]
        try:
            leads.append(lookup[title])
        except KeyError:
            if default is None:
                raise KeyError(f"Title '{title}' not found in {jsonl_path}") from None
            leads.append(default)

    return leads











    

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Specify embedding type.')
    parser.add_argument('--embedding', type=str, default='TF-IDF', help='The type of embedding to use')
    parser.add_argument('--num_samples', type=int, default=NUMBER_OF_SAMPLE, help='Number of samples to use')
    parser.add_argument('--metric', type=str, default='cosine', help='The metric to use for clustering')
    parser.add_argument('--PCA', action="store_true", default=False, help='Apply PCA')
    parser.add_argument('--g', action="store_true", default=False, help='Whether to plot PCA explained variance')
    args = parser.parse_args()
    main(args)