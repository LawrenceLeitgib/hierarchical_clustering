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

def build_Linkage_Matrix(tree_set,num_samples):
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
                    #print("biggest[0][0]",biggest[0][0])
                    #print("linkage_matrix",linkage_matrix)
                    #print(len(offsetArray))
                    
                    l1=biggest[0][0]+(num_samples +offsetArray[biggest[0][0]]if biggest[0][1] else 0)
                l2=biggest[j][0]+(num_samples+offsetArray[biggest[j][0]] if biggest[j][1] else 0)

                for el in linkage_matrix:
                    if(l1 == el[0] or l2 == el[1] or l1 == el[1] or l2 == el[0]):
                        print("l1",l1)
                        print("l2",l2)
                        print("test----------------------------------------")
                        break


                linkage_matrix.append([l1, l2, highest_hight+1.0, len(e)])
                distanceArray[i]=highest_hight+1
                index+=1
           
            for j in range(i,len(tree_set)):
                offsetArray[j]+=(len(biggest)-2)




    print(linkage_matrix)
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

    colorList=generate_distinct_colors(len(set(categories)))
    with open('resources/categories_name_map.json') as json_file:
        data = json.load(json_file)


    i=0
    for s in categories:
        if s not in color_map:
            color_map[s] = colorList[i]
            i+=1
        
        colors.append(color_map[s])
        names.append(data[s][0])


    return colors,names

def assign_colors_sub_categories(categories):
    #load the arxiv_categories_map.json
    with open('resources/categories_name_map.json') as json_file:
        data = json.load(json_file)
    colorList=generate_distinct_colors(len(parent_category_to_id))
    print(categories)
    colors = []
    names = []
    for subCat in categories:
        info=data[subCat]
        color=colorList[parent_category_to_id[info[1]]]
        colors.append(color)
        names.append(info[0])

    return colors,names


def plot_tree(tree,num_samples,n,dataset,isSet=True):
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
    if(dataset == "wikipedia"):
      

        names2=np.load(f"wikipedia_labels/wikivitals_names_{num_samples}.npy",allow_pickle=True)


        image = visualize_dendrogram(np.array(T_linkage), names=names2, rotate=True, scale=4, n_clusters=4)

        soup = BeautifulSoup(image, "xml")
        svg_tag = soup.find("svg")

        # Add a white background rectangle as the first element
        if svg_tag:
            bg_rect = soup.new_tag("rect", x="0", y="0", width="100%", height="100%", fill="white")
            svg_tag.insert(0, bg_rect)
        # Save modified SVG
        with open("dendrogram.svg", "w", encoding="utf-8") as f:
            f.write(str(soup))

    

        # Show image
        #plt.imshow(img)
        if(isSet):
            with open("out/T_Star.svg", "w", encoding="utf-8") as f:
                f.write(str(soup))
        else:
            with open("out/T_Binary.svg", "w", encoding="utf-8") as f:
                f.write(str(soup))
        return
    else:

        typeList="map_" if n==num_samples else "list_"

        category_list = np.load("categories_list/categories_"+typeList+str(num_samples)+".npy",allow_pickle=True)

        if(n==num_samples):
            color_list,category_names = assign_colors(category_list)
        else:
            color_list,category_names = assign_colors_sub_categories(category_list)



        ax = plt.gca()
        # Get the leaf order from the dendrogram
        leaf_order = dn['leaves']
        # Get the color of each leaf
        leaf_colors = [color_list[int(i)] for i in leaf_order]
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
        if n != num_samples:
            legend_handles = []
            allcolors = generate_distinct_colors(len(parent_category_to_id))
            for key, value in parent_category_to_id.items():            
                patch = mpatches.Patch(color=allcolors[value], label=key)
                legend_handles.append(patch)

            # Create the legend *outside* the plot
            plt.legend(
                handles=legend_handles,
                title='Parent Categories',
                facecolor='white',
                edgecolor='black',
                bbox_to_anchor=(0.5, 1.05),  # Position the legend at the top center
                loc='center',  # Align the legend horizontally
                ncol=1  # Number of columns in the legend
        )
        # Adjust layout to make room for the legend
        plt.tight_layout(rect=[0, 0, 1, 0.95])  # Adjust the rect to leave space for the lables


    plt.title('Dendrogram of Clusters')

    plt.draw()  # Ensure the updates are shown.
    if(isSet):
        plt.savefig('out/T_Star.pdf')
    else:
        plt.savefig('out/T_Binary.pdf')


