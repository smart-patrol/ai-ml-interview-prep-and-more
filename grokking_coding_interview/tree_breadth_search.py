from typing import List
from collections import deque


class TreeNode:
    def __init__(self,val) -> None:
        self.val = val
        self.left = None
        self.right = None
        self.next = None
    # tree traversal using 'next' pointer
    def print_tree(self):
        print("Traversal using 'next' pointer: ", end='')
        current = self
        while current:
            print(str(current.val) + " ", end='')
            current = current.next

def connect_all_siblings(root:'TreeNode') -> 'TreeNode':
    if root is None:
        return
    
    queue  = deque()
    queue.append(root)
    current_node, previous_node = None, None
    while queue:
        current_node = queue.popleft()
        if previous_node:
            previous_node.next = current_node
        previous_node = current_node

        if current_node.left:
            queue.append(current_node.left)
        if current_node.right:
            queue.append(current_node.right)

    return root

def main():
    root = TreeNode(12)
    root.left = TreeNode(7)
    root.right = TreeNode(1)
    root.left.left = TreeNode(9)
    root.right.left = TreeNode(10)
    root.right.right = TreeNode(5)
    connect_all_siblings(root)
    root.print_tree()

main()

# level order traversal using 'next' pointer
def print_level_order(self):
    nextLevelRoot = self
    while nextLevelRoot:
        current = nextLevelRoot
        nextLevelRoot = None
        while current:
            print(str(current.val) + " ", end='')
            if not nextLevelRoot:
                if current.left:
                    nextLevelRoot = current.left
                elif current.right:
                    nextLevelRoot = current.right
            current = current.next
        print()

def connect_level_order_siblings(root:'TreeNode') -> 'TreeNode':
    """
    Given a binary tree, connect each node with its level order successor. 
    The last node of each level should point to a null node.
    """
    if root is None:
        return
    
    queue = deque()
    queue.append(root)
    while queue:
        previous_node = None
        level_size = len(queue)
        # connect all nodes of this level
        for _ in range(level_size):
            current_node = queue.popleft()
            if previous_node:
                previous_node.next = current_node
            previous_node = current_node

            # insert the children of the current node into the queue
            if current_node.left:
                queue.append(current_node.left)
            if current_node.right:
                queue.append(current_node.right)
    
    return root

# def main():
#   root = TreeNode(12)
#   root.left = TreeNode(7)
#   root.right = TreeNode(1)
#   root.left.left = TreeNode(9)
#   root.right.left = TreeNode(10)
#   root.right.right = TreeNode(5)
#   connect_level_order_siblings(root)

# main()


# class TreeNode:
#     def __init__(self,val) -> None:
#         self.val = val
#         self.left = None
#         self.right = None


def find_successor(root:'TreeNode', key:int) -> int:
    """
    Given a binary tree and a node, find the level order successor of the given node in the tree. The level order successor is the node that appears right after the given node in the level order traversal.
    """

    queue = deque()
    queue.append(root)
    while queue:
        current_node = queue.popleft()
        if current_node.left:
            queue.append(current_node.left)
        if current_node.right:
            queue.append(current_node.right)
    
        if current_node.val == key:
            break
    return queue[0] if queue else None

        
# def main():
#   root = TreeNode(12)
#   root.left = TreeNode(7)
#   root.right = TreeNode(1)
#   root.left.left = TreeNode(9)
#   root.right.left = TreeNode(10)
#   root.right.right = TreeNode(5)
#   result = find_successor(root, 12)
#   if result:
#     print(result.val)
#   result = find_successor(root, 9)
#   if result:
#     print(result.val)

# main()


def find_minimum_depth(root:'TreeNode') -> int:
    """
    Find the minimum depth of a binary tree. 
    The minimum depth is the number of nodes along the shortest path from the 
    root node to the nearest leaf node.
    """
    if root is None:
        return 0
    
    queue = deque()
    queue.append(root)
    minimum_tree_depth = 0

    while queue:
        minimum_tree_depth += 1
        level_size = len(queue)
        for _ in range(level_size):
            current_node = queue.popleft()

            # check if this is a leaf node
            if current_node.left is None and current_node.right is None:
                return minimum_tree_depth
            # insert the children of the current node into the queue
            if current_node.left:
                queue.append(current_node.left)
            if current_node.right:
                queue.append(current_node.right)
    
    return minimum_tree_depth


# def main():
#   root = TreeNode(12)
#   root.left = TreeNode(7)
#   root.right = TreeNode(1)
#   root.right.left = TreeNode(10)
#   root.right.right = TreeNode(5)
#   print("Tree Minimum Depth: " + str(find_minimum_depth(root)))
#   root.left.left = TreeNode(9)
#   root.right.left.left = TreeNode(11)
#   print("Tree Minimum Depth: " + str(find_minimum_depth(root)))


# main()

def traverse(root):
    result = []

    if root is None:
        return result
    
    queue = deque()
    queue.append(root)

    while queue:
        levelSize = len(queue)
        currentLevel = []
        for _ in range(levelSize):
            currentNode =queue.popleft()
            currentLevel.append(currentNode.val)
            if currentNode.left:
                queue.append(currentNode.left)
            if currentNode.right:
                queue.append(currentNode.right)
        result.append(currentLevel)

    return result

# def main():
#   root = TreeNode(12)
#   root.left = TreeNode(7)
#   root.right = TreeNode(1)
#   root.left.left = TreeNode(9)
#   root.right.left = TreeNode(10)
#   root.right.right = TreeNode(5)
#   print("Level order traversal: " + str(traverse(root)))

# main()