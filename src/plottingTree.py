import networkx as nx
from scipy.cluster.hierarchy import dendrogram, linkage

import matplotlib.pyplot as plt
import numpy as np
import random

def build_Linkage_Matrix(tree_set):

    #remove from the tree set each element of size 1
    length =  int(sorted([x for x in tree_set if len(x) == 1],key=lambda x: int(x[0]), reverse=True)[0][0])+1
    #print(length)
    tree_set = [x for x in tree_set if len(x) > 1]
    #print("----------------------------")
    #print(tree_set)
    linkage_matrix = []
    if(len(tree_set)==1):
        e=tree_set[0]
        linkage_matrix.append([int(e[0]), int(e[1]), 1.0, 2])
        for i in range(2,len(e)):
            linkage_matrix.append([len(e)+i-2, int(e[i]), 1.0, i+1])
        print(linkage_matrix)
        return linkage_matrix

    offsetArray = [0]*len(tree_set)
    distanceArray = [0]*len(tree_set)
    index=length-1
    for i in range(len(tree_set)):
        e = sorted(tree_set[i])
        if(len(e) ==2):
            linkage_matrix.append([int(e[0]), int(e[1]), 1.0, len(e)])
            distanceArray[i]=1
            index+=1
        else:
            subset_list=[]
            for j in range(i):
                if(set(tree_set[j]).issubset(set(e))):
                    subset_list.append(j)

            subset_list.sort(key=lambda x: len(tree_set[x]), reverse=True)

            #the first element is the index of the subset and the second element indicate if the subset is a composed cluster
            biggest = [(subset_list[0],True)]
            for j in range(1,len(subset_list)):
                is_subset = False
                for b in biggest:
                    if(set(tree_set[subset_list[j]]).issubset(set(tree_set[b[0]]))):
                        is_subset = True
                        break
                if(not is_subset):
                    biggest.append((subset_list[j],True))
            
            diff = set(e)
            for b in biggest:
                diff = diff - set(tree_set[b[0]])
            for el in diff:
                biggest.append((int(el),False))
            print("------------")
            highest_hight=1
            for b in biggest:
               if(b[1]):
                   if(distanceArray[b[0]]>highest_hight):
                        highest_hight=distanceArray[b[0]]
          
            for j in range(1,len(biggest)):
                if(j>1):
                    l1=index
                else :
                    l1=biggest[0][0]+(length if biggest[0][1] else 0)+offsetArray[biggest[0][0]]
                l2=biggest[j][0]+(length+offsetArray[biggest[j][0]] if biggest[j][1] else 0)


                linkage_matrix.append([l1, l2, highest_hight+1, len(e)])
                distanceArray[i]=highest_hight+1
                index+=1
           
            for j in range(i,len(tree_set)):
                offsetArray[j]+=(len(biggest)-2)




    #print(linkage_matrix)
    return linkage_matrix



def generate_random_color():
    """Generate a random RGB color."""
    return f'#{random.randint(0, 255):02x}{random.randint(0, 255):02x}{random.randint(0, 255):02x}'

def assign_colors(strings):
    """Assign a unique random color to each unique string."""
    color_map = {}
    result = []

    for s in strings:
        if s not in color_map:
            color_map[s] = generate_random_color()
        result.append(color_map[s])
    
    return result


def plot_tree(tree_set):
    """
    Plot the tree using networkx and matplotlib.
    """
    T_linkage = build_Linkage_Matrix(tree_set)
    #color each singleton cluster base on the category list
    length =  int(sorted([x for x in tree_set if len(x) == 1],key=lambda x: int(x[0]), reverse=True)[0][0])+1
    category_list = np.load("categories_list/categories_list_"+str(length)+".npy",allow_pickle=True)
    print(category_list)
    color_list = assign_colors(category_list)
    print(color_list)


    fig = plt.figure(figsize=(35, 15))

    dn = dendrogram(T_linkage)

    ax = plt.gca()
    x_labels = ax.get_xticklabels()

    x_labels = sorted(x_labels, key=lambda x: int(x.get_text()))

    


    for lbl, color in zip(x_labels, color_list):
        lbl.set_color(color)  # Set each label to a different color
    plt.savefig('T*.png') 


