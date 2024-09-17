class TreeNode:
    def __init__(self, val):
        self.val = val
        self.left = None
        self.right = None


# Insert a new node and return the root of the BST.
def insert(root, val):
    if not root:
        return TreeNode(val)

    if val > root.val:
        root.right = insert(root.right, val)
    elif val < root.val:
        root.left = insert(root.left, val)
    return root


# Return the minimum value node of the BST.
def minValueNode(root):
    curr = root
    while curr and curr.left:
        curr = curr.left
    return curr


# Remove a node and return the root of the BST.
def remove(root, val):
    # Base case: if the tree is empty
    if not root:
        return None
    # Base case: if the node to be deleted is not found
    if val > root.val:
        root.right = remove(root.right, val)
    elif val < root.val:
        root.left = remove(root.left, val)
    # Recursive cases: node with children or no children
    else:
        # cases for child is null
        if not root.left:
            return root.right
        elif not root.right:
            return root.left
        # other case when the node has two children that need to be replaced
        else:
            # Replace the value with the smallest value in the right subtree.
            minNode = minValueNode(root.right)
            root.val = minNode.val
            root.right = remove(root.right, minNode.val)
    return root
