#creat a class where Node have a parent and the list of elements and children

class Node:
    def __init__(self, data):
        self.data = data
        self.parent = None
        self.children = []

    def add_child(self, child):
        child.parent = self
        self.children.append(child)
    


def tree_set_to_tree(tree_set):
    tree_set = sorted(tree_set, key=lambda x: len(x),reverse=True)
    root = Node(tree_set[0])

    for i in range(1,len(tree_set)):
        tree_to_tree_set_recursive(root,tree_set[i])

    #print_tree(root,0)
    return root

def tree_to_tree_set_recursive(node,element):
    isGrandChild=False
    for c in node.children:
        if(set(element).issubset(set(c.data))):
            tree_to_tree_set_recursive(c,element)
            isGrandChild=True
    if(not isGrandChild):
        node.add_child(Node(element)) 


def print_tree(root,level):
    print("-----------------------------")
    print("     " * level + str(root.data))
    for c in root.children:
        print_tree(c,level+1)