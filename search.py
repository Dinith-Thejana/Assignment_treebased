import sys




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
def parse_file(filename):
    """Reads the file and builds the graph correctly."""
    graph = Graph()
    with open(filename, 'r') as file:
        lines = [line.strip() for line in file.readlines() if line.strip()]  # Remove empty lines

    mode = None
    origin = None
    destinations = []

    i = 0
    while i < len(lines):
        line = lines[i]

        if line.startswith("Nodes:"):
            mode = "nodes"
        elif line.startswith("Edges:"):
            mode = "edges"
        elif line.startswith("Origin:"):
            # Move to next line for origin value
            i += 1
            while i < len(lines) and lines[i] == "":
                i += 1
            if i < len(lines):
                origin = int(lines[i])
            else:
                print("Error: Missing origin value")
                sys.exit(1)
        elif line.startswith("Destinations:"):
            # Move to next line for destinations value
            i += 1
            while i < len(lines) and lines[i] == "":
                i += 1
            if i < len(lines):
                destinations = list(map(int, lines[i].replace(" ", "").split(";")))
            else:
                print("Error: Missing destinations value")
                sys.exit(1)
        elif mode == "nodes":
            try:
                node_id, coords = line.split(":")
                x, y = map(int, coords.strip().strip("()").replace(" ", "").split(","))
                graph.add_node(int(node_id), x, y)
            except ValueError:
                print(f"Skipping invalid node format: {line}")
        elif mode == "edges":
            try:
                edge_data, cost = line.split(":")
                start, end = map(int, edge_data.strip("()").split(","))
                graph.add_edge(start, end, int(cost))
            except ValueError:
                print(f"Skipping invalid edge format: {line}")

        i += 1

    if origin is None:
        print("Error: No valid 'Origin' specified in the file.")
        sys.exit(1)
    if not destinations:
        print("Error: No valid 'Destinations' specified in the file.")
        sys.exit(1)

    return graph, origin, destinations

def dfs(graph, start, goal):
    stack = [(start, [start])]
    visited = set()
    
    while stack:
        node, path = stack.pop()
        if node in visited:
            continue
        visited.add(node)

        if node == goal:
            return path
        
        for neighbor in graph.edges[node]:
            if neighbor not in visited:
                stack.append((neighbor, path + [neighbor]))
    
    return None
# Custom Search 1 - Iterative Deepening DFS
def cus1(graph, start, goal):
    def dls(node, goal, depth, path):
        if depth == 0 and node == goal:
            return path
        if depth > 0:
            for neighbor in graph.edges[node]:
                if neighbor not in path:
                    result = dls(neighbor, goal, depth - 1, path + [neighbor])
                    if result:
                        return result
        return None

    depth = 0
    while True:
        result = dls(start, goal, depth, [start])
        if result:
            return result
        depth += 1