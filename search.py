class Graph:
    def __init__(self):
        self.nodes = {}  # Stores (x, y) coordinates of nodes
        self.edges = {}  # Stores adjacency list with costs

    def add_node(self, node, x, y):
        self.nodes[node] = (x, y)
        self.edges[node] = {}

    def add_edge(self, start, end, cost):
        self.edges[start][end] = cost
        self.edges[end][start] = cost  # Assuming undirected graph
