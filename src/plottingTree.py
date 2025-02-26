import networkx as nx
import matplotlib.pyplot as plt

def build_graph(tree_set):
    """
    Build a graph from a set of sets representing a non-binary tree.
    """
    G = nx.DiGraph()
    G2 = nx.DiGraph()


   
    # Root node (the largest set)
    root = tuple(sorted(max(tree_set, key=len)))
    to_handle=[]
    for s in tree_set:
            if(len(s) == 1):
                G.add_node(s)
                G2.add_node(s)
            else :
                to_handle.append(tuple(sorted(s)))

    while len(to_handle) > 0:
        smallest = tuple(sorted(min(to_handle, key=len)))    
        G.add_node(smallest)
        G2.add_node(smallest)

        to_remove = []
        for node in G2.nodes():
            if node != smallest:
                if set(node).issubset(smallest):
                    G.add_edge(smallest, node)
                    to_remove.append(node)
        for node in to_remove:
            G2.remove_node(node)

       
        to_handle.remove(smallest)


    
    #print(G)

    return G, root

def plot_tree(G, root):
    """
    Plot the tree using networkx and matplotlib.
    """
    pos = hierarchy_pos(G, root)
    #print(pos)
    plt.figure(figsize=(12, 8))
    nx.draw(G, pos, with_labels=True, arrows=False, node_size=3000, node_color='skyblue', font_size=10)
    plt.title("Non-Binary Tree Visualization")
    plt.savefig('T*.png') 

def hierarchy_pos(G, root, width=1.0, vert_gap=0.2, pos=0.5):
    """
    Position nodes in a hierarchy layout (supports non-binary trees).
    """
    def _hierarchy_pos(G, root, pos, width, vert_gap,allPos):
        children = list(G.successors(root))
        
        root_label = root

        if len(children) != 0:
            dx = width / len(children)
            leftmost =pos - width/2
            for child in children:
                allPos = _hierarchy_pos(G, child,leftmost,width/len(children), vert_gap+0.2,allPos)
                leftmost += dx
        allPos[root_label] = (pos, vert_gap)
        #print(allPos)
        return allPos

    return _hierarchy_pos(G, root, pos, width, vert_gap, {})

