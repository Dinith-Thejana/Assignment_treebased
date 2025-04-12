import sys
import heapq
import math

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

    def heuristic(self, node, goal):
        x1, y1 = self.nodes[node]
        x2, y2 = self.nodes[goal]
        return math.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2) 

def parse_file(filename):
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
        elif line.startswith("Origin:"): # Move to next line for origin value
            i += 1
            while i < len(lines) and lines[i] == "":
                i += 1
            if i < len(lines):
                origin = int(lines[i])
            else:
                print("Error: Missing origin value")
                sys.exit(1)
        elif line.startswith("Destinations:"): # Move to next line for destinations value
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
            return path, visited
        
        for neighbor in sorted(graph.edges[node]):
            if neighbor not in visited:
                stack.append((neighbor, path + [neighbor]))
    
    return None, visited

def bfs(graph, start, goal):
    queue = [(start, [start])]
    visited = set()
    
    while queue:
        node, path = queue.pop(0)
        if node in visited:
            continue
        visited.add(node)

        if node == goal:
            return path,visited
        
        for neighbor in sorted(graph.edges[node]):
            if neighbor not in visited:
                queue.append((neighbor, path + [neighbor]))

    return None, visited

def gbfs(graph, start, goal):
    pq = [(graph.heuristic(start, goal), start, [start])]
    visited = set()

    while pq:
        _, node, path = heapq.heappop(pq)
        if node in visited:
            continue
        visited.add(node)

        if node == goal:
            return path, visited

        for neighbor in sorted(graph.edges[node]):
            if neighbor not in visited:
                heapq.heappush(pq, (graph.heuristic(neighbor, goal), neighbor, path + [neighbor]))

    return None, visited

def a_star(graph, start, goal):
    pq = [(0, start, [start], 0)]  # (f, node, path, g)
    visited = set()
    cost_dict = {start: 0}  

    while pq:
        _, node, path, g = heapq.heappop(pq)
        if node in visited:
            continue
        visited.add(node)

        if node == goal:
            return path, g, visited 

        for neighbor in sorted(graph.edges[node]):
            new_g = g + graph.edges[node][neighbor]
            if neighbor not in cost_dict or new_g < cost_dict[neighbor]:
                cost_dict[neighbor] = new_g
                f = new_g + graph.heuristic(neighbor, goal)
                heapq.heappush(pq, (f, neighbor, path + [neighbor], new_g))

    return None, float('inf'), visited

# Custom Search 1 
def cus1(graph, start, goal):
    visited = set()

    def dls(node, goal, depth, path):
        visited.add(node)
        if depth == 0 and node == goal:
            return path
        if depth > 0:
            for neighbor in sorted(graph.edges[node]):
                if neighbor not in path:
                    result = dls(neighbor, goal, depth - 1, path + [neighbor])
                    if result:
                        return result
        return None

    depth = 0
    while True:
        visited.clear()
        result = dls(start, goal, depth, [start])
        if result:
            return result, visited
        depth += 1

# Custom Search 2
def cus2(graph, start, goal):
    pq = [(0, start, [start])]  
    visited = set()

    while pq:
        cost, node, path = heapq.heappop(pq)
        if node in visited:
            continue
        visited.add(node)

        if node == goal:
            return path, cost, visited

        for neighbor in graph.edges[node]:
            heapq.heappush(pq, (cost + graph.edges[node][neighbor], neighbor, path + [neighbor]))

    return None, float('inf'), visited

def calculate_path_cost(graph, path):
    if not path or len(path) < 2:
        return 0
    cost = 0
    for i in range(len(path) - 1):
        cost += graph.edges[path[i]][path[i + 1]]
    return cost

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python search.py <filename> <method>")
        sys.exit(1)

    filename = sys.argv[1]
    method = sys.argv[2].upper()

    graph, origin, destinations = parse_file(filename)

    best_path = None
    best_cost = float('inf')
    best_goal = None
    visited_count = 0

    for goal in destinations:
        if method == "DFS":
            path, visited = dfs(graph, origin, goal)
            cost = calculate_path_cost(graph, path)
        elif method == "BFS":
            path, visited = bfs(graph, origin, goal)
            cost = calculate_path_cost(graph, path)
        elif method == "GBFS":
            path, visited = gbfs(graph, origin, goal)
            cost = calculate_path_cost(graph, path)
        elif method == "AS":
            path, cost, visited = a_star(graph, origin, goal)
        elif method == "IDDFS":
            path, visited = cus1(graph, origin, goal)
            cost = calculate_path_cost(graph, path)
        elif method == "UCS":
            path, cost, visited = cus2(graph, origin, goal)
        else:
            print(f"Invalid method: {method}")
            sys.exit(1)

        if path and cost < best_cost:
            best_path = path
            best_cost = cost
            best_goal = goal
            visited_count = len(visited)

    if best_path:
        print(f"{filename} {method}")
        print(f"Goal: {best_goal}, Nodes Expanded: {visited_count}, Cost: {best_cost}")
        print("Path:", " -> ".join(map(str, best_path)))
    else:
        print(f"No valid path found from {origin} to any destination using {method}.")


    