from scipy.cluster.hierarchy import dendrogram, linkage
import matplotlib.colors as mcolors
import matplotlib.patches as mpatches
from sknetwork.visualization import visualize_dendrogram

from sknetwork.data import load_netset
from sknetwork.ranking import PageRank, top_k
from bs4 import BeautifulSoup

import matplotlib.image as mpimg
from io import BytesIO
import cairosvg


import matplotlib.pyplot as plt
import numpy as np
import json
parent_category_to_id = {
    "Computer Science": 0,
    "Economics": 1,
    "Electrical Engineering and Systems Science": 2,
    "Mathematics": 3,
    "Physics": 4,
    "Quantitative Biology": 5,
    "Quantitative Finance": 6,
    "Statistics": 7
}

wikipedia_categoriy_to_id ={'Arts': 0, 
                            'Biological and health sciences': 1, 
                            'Everyday life': 2, 
                            'Geography': 3,
                            'History': 4, 
                            'Mathematics': 5, 
                            'People': 6, 
                            'Philosophy and religion': 7, 
                            'Physical sciences': 8,
                            'Society and social sciences': 9,
                            'Technology': 10}


def build_Linkage_Matrix(tree_set,num_samples ):
    tree_set = [x for x in tree_set if len(x) > 1]
    linkage_matrix = []

    offsetArray = [0]*len(tree_set)
    distanceArray = [0]*len(tree_set)
    index=num_samples-1
    for i in range(len(tree_set)):
        e = sorted(tree_set[i])
        if(len(e) ==2):
            linkage_matrix.append([int(e[0]), int(e[1]), 1.0, 2])
            distanceArray[i]=1
            index+=1
        else:
            subset_list=[]
            for j in range(i):
                if(set(tree_set[j]).issubset(set(e))):
                    subset_list.append(j)

            subset_list.sort(key=lambda x: len(tree_set[x]), reverse=True)

            #the first element is the index of the subset and the second element indicate if the subset is a composed cluster
            if(len(subset_list)>0):
                biggest = [(subset_list[0],True)]
                for j in range(1,len(subset_list)):
                    is_subset = False
                    for b in biggest:
                        if(set(tree_set[subset_list[j]]).issubset(set(tree_set[b[0]]))):
                            is_subset = True
                            break
                    if(not is_subset):
                        biggest.append((subset_list[j],True))
            else:
                biggest = []
            
            diff = set(e)
            for b in biggest:
                diff = diff - set(tree_set[b[0]])
            for el in diff:
                biggest.append((int(el),False))
            highest_hight=0
            for b in biggest:
               if(b[1]):
                   if(distanceArray[b[0]]>highest_hight):
                        highest_hight=distanceArray[b[0]]
          
            for j in range(1,len(biggest)):
                if(j>1):
                    l1=index
                else :
        
                    
                    l1=biggest[0][0]+(num_samples +offsetArray[biggest[0][0]]if biggest[0][1] else 0)
                l2=biggest[j][0]+(num_samples+offsetArray[biggest[j][0]] if biggest[j][1] else 0)


                linkage_matrix.append([l1, l2, highest_hight+1.0, len(e)])
                distanceArray[i]=highest_hight+1
                index+=1
           
            for j in range(i,len(tree_set)):
                offsetArray[j]+=(len(biggest)-2)




    return linkage_matrix




def generate_distinct_colors(n):
    colors = []
    for i in range(n):       
        rgb = mcolors.hsv_to_rgb([i/n, 1, 0.6])
        colors.append(tuple(rgb))  # Convert to tuple for easy use
    return colors

def assign_colors(categories):
    """Assign a unique random color to each unique string."""
    color_map = {}
    colors = []
    names = []
    true_labels = []

    colorList=generate_distinct_colors(len(set(categories)))
    with open('resources/categories_name_map.json') as json_file:
        data = json.load(json_file)


    i=0
    for s in categories:
        if s not in color_map:
            color_map[s] = (colorList[i],i)
            i+=1
        
        colors.append(color_map[s][0])
        names.append(data[s][0])
        true_labels.append(color_map[s][1])


    return colors,names,true_labels

def assign_colors_sub_categories(categories):
    #load the arxiv_categories_map.json
    with open('resources/categories_name_map.json') as json_file:
        data = json.load(json_file)
    colorList=generate_distinct_colors(len(parent_category_to_id))
    colors = []
    names = []
    trueLabels = []
    for subCat in categories:
        info=data[subCat]
        color=colorList[parent_category_to_id[info[1]]]
        colors.append(color)
        names.append(info[0])
        trueLabels.append(parent_category_to_id[info[1]])

    return colors,names,trueLabels
def assign_colors_wikipedia(categories):

    colorList=generate_distinct_colors(len(wikipedia_categoriy_to_id))
    colors = []
    names = []
    true_labels = []
    for el in categories:
        names.append(el[0])
        color=colorList[wikipedia_categoriy_to_id[el[1]]]
        colors.append(color)
        true_labels.append(wikipedia_categoriy_to_id[el[1]])

    return colors,names,true_labels


import higra as hg

def sets_to_higra_tree(list_of_sets):
    # 1) sort nodes by size ascending (so leaves first)
    #    tie-break by converting to sorted tuple for reproducibility
 
        
    nodes = list(list_of_sets)
    #order = sorted(range(len(nodes)),
    #               key=lambda i: (len(nodes[i]), tuple(sorted(nodes[i]))))
    #nodes = [nodes[i] for i in order]

    n = len(nodes)
    parents = np.empty(n, dtype=int)

    # 2) for each node i, find the minimal j>i s.t. nodes[i] ⊆ nodes[j]
    for i in range(n):
        supers = [j for j in range(i+1, n)
                  if set(nodes[i]).issubset(set(nodes[j]))]
        if supers:
            # pick the one with smallest size (they’re already sorted by size)
            parents[i] = supers[0]
        else:
            # no superset ⇒ this must be the root
            parents[i] = i

    # 3) ensure root points to itself
    root = np.argmax([len(s) for s in nodes])
    parents[root] = root

    # 4) construct the Higra tree
    tree = hg.Tree(parents)
    return tree


def plot_tree(tree,num_samples,n,dataset,isSet=True,optionalSet=None,name="T.pdf"):
    """
    Plot the tree using networkx and matplotlib.
    """
    if(isSet):
        T_linkage = build_Linkage_Matrix(tree,n)
    else:
        T_linkage = tree

    fig = plt.figure(figsize=(12, 29))
    dn = dendrogram(T_linkage,orientation='left')


    #color each singleton cluster base on the category list



    if(dataset=="wikipedia"):
        category_list = np.load(f"wikipedia_labels/wikivitals_names_{num_samples}.npy",allow_pickle=True)
    else:
        typeList="map_" if n==num_samples else "list_"
        category_list = np.load("categories_list/categories_"+typeList+str(num_samples)+".npy",allow_pickle=True)
       
    true_label=None
    if(dataset=="abstracts"):
        color_list,category_names,true_label = assign_colors(category_list)
    elif(dataset=="categories"):
        color_list,category_names,true_label = assign_colors_sub_categories(category_list)
    elif(dataset=="wikipedia"):
        color_list,category_names,true_label = assign_colors_wikipedia(category_list)



    ax = plt.gca()
    # Get the leaf order from the dendrogram
    leaf_order = dn['leaves']
    # Get the color of each leaf


    leaf_colors = [color_list[int(i)] for i in leaf_order]

    if(dataset=="wikipedia"):
        category_names = [category_list[int(i)][1] for i in leaf_order]
        category_list = [category_list[int(i)][0] for i in leaf_order]
    else:
        category_list = [category_list[int(i)] for i in leaf_order]
        category_names = [category_names[int(i)] for i in leaf_order]
    # Set the color of each leaf    
    for leaf_patch, color in zip(ax.get_yticklabels(), leaf_colors):
        leaf_patch.set_color(color)


    new_labels = [cat+"||"+l for cat,l in zip(category_names,category_list)]



    # Apply the new labels
    ax.set_yticklabels(new_labels)

    #change the font size of the lables
    for label in ax.get_yticklabels():
        label.set_fontsize(12)
    



    for coll in ax.collections:
        coll.set_color('black')


    #use parent_category_to_id and the color_list to create a legend
    if dataset=="categories" or dataset=="wikipedia":
        legend_handles = []
        dictionary = parent_category_to_id if dataset=="categories" else wikipedia_categoriy_to_id
        allcolors = generate_distinct_colors(len(dictionary))
        for key, value in dictionary.items():            
            patch = mpatches.Patch(color=allcolors[value], label=key)
            legend_handles.append(patch)

        # Create the legend *outside* the plot
        plt.legend(
            handles=legend_handles,
            fontsize="x-large",
            title='Labels',
            facecolor='white',
            edgecolor='black',
            bbox_to_anchor=(0.5, 1.15),  # Position the legend at the top center
            loc='center',  # Align the legend horizontally
            ncol=1  # Number of columns in the legend

    )
    # Adjust layout to make room for the legend
    plt.tight_layout(rect=[0, 0, 1, 0.95])  # Adjust the rect to leave space for the lables


    plt.title('Dendrogram of Clusters')

    plt.draw()  # Ensure the updates are shown.

    plt.savefig('out/'+name)
   
    #compute purity
    if(isSet):
        treeH = sets_to_higra_tree(tree)
        #print(tree)
        #print(true_label)
        purity = hg.dendrogram_purity(treeH, np.array(true_label))
        print("Dendrogram purity of T_star:", purity)
    else:
        treeH = sets_to_higra_tree(optionalSet)
        purity = hg.dendrogram_purity(treeH, np.array(true_label))
        print("Dendrogram purity of T_binary:", purity)
    



