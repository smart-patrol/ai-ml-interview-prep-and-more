from collections import defaultdict


class Graph:
    # constructor
    def __init__(self, vertices):
        self.graph = defaultdict(list)
        self.vertices = vertices

    def addEdge(self, u, v):
        self.graph[u].append(v)


def helper(my_graph, current_node, visited):
    # mark the current node as visited and print it
    if visited[current_node] == False:
        visited[current_node] = True
        print(current_node)

    # recur for all the vertices adjacent to current node
    for i in my_graph.graph[current_node]:
        if visited[i] == False:
            helper(my_graph, i, visited)


def dfs(my_graph):
    # initially all vertices are marked not visited
    visited = [False] * (my_graph.vertices)
    helper(my_graph, 0, visited)  # call the helper function starting from node 0


# Driver code

# Create a graph given in the above diagram
myGraph = Graph(6)
myGraph.addEdge(0, 1)
myGraph.addEdge(1, 2)
myGraph.addEdge(1, 3)
myGraph.addEdge(2, 4)
myGraph.addEdge(3, 4)
myGraph.addEdge(3, 5)

print("DFS Graph Traversal")
dfs(myGraph)
