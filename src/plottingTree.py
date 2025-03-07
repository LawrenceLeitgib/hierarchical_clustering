import networkx as nx
from scipy.cluster.hierarchy import dendrogram, linkage

import matplotlib.pyplot as plt

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

    distance = 1.0
    offsetArray = [0]*len(tree_set)
    index=length-1
    for i in range(len(tree_set)):
        e = sorted(tree_set[i])
        if(len(e) ==2):
            linkage_matrix.append([int(e[0]), int(e[1]), distance/(len(tree_set)), len(e)])
            distance+=1
            index+=1
        else:
            subset_list=[]
            for j in range(i):
                if(set(tree_set[j]).issubset(set(e))):
                    subset_list.append(j)

            subset_list.sort(key=lambda x: len(tree_set[x]), reverse=True)
            #the first element is the index of the subset and the second element indicate if the subset is nmot a single element
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
            #print(diff)
            #print("cccccccccccccccccccccccccccccccccccccccccccccccc")
            #print(biggest)
            #biggest.sort(key=lambda x: int(x[0])+(length if x[1] else 0),reverse=True)
            #print(subset_list)
            #print(biggest)
            #print(offsetArray)
           # print("$$$$$$$$$$$$$$$$$$$$$$$")
            #print(biggest)
            for j in range(1,len(biggest)):
                if(j>1):
                    l1=index
                else :
                    l1=biggest[0][0]+(length if biggest[0][1] else 0)+offsetArray[biggest[0][0]]
                l2=biggest[j][0]+(length if biggest[j][1] else 0)+(offsetArray[biggest[j][0]] if biggest[j][1] else 0)
                linkage_matrix.append([l1, l2, distance/(len(tree_set)), len(e)])
                index+=1
           
            for j in range(i,len(tree_set)):
                offsetArray[j]+=(len(biggest)-2)
            distance+=1




    #print(linkage_matrix)
    return linkage_matrix

def plot_tree(tree_set):
    """
    Plot the tree using networkx and matplotlib.
    """
    T_linkage = build_Linkage_Matrix(tree_set)
    #print(pos)
    fig = plt.figure(figsize=(35, 15))
    dn = dendrogram(T_linkage)
    plt.savefig('T*.png') 


