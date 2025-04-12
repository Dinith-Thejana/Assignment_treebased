# COS30019 – Assignment 2A: Tree-Based Search

## 📌 Overview
This project solves the **Route Finding Problem** using six different tree-based search algorithms. The goal is to find the optimal or valid path from a starting node (origin) to one or more destination nodes on a 2D graph.

---

## 🔍 Search Algorithms Implemented

| Algorithm | Type        | Description |
|-----------|-------------|-------------|
| DFS       | Uninformed  | Depth-First Search – explores as far as possible before backtracking |
| BFS       | Uninformed  | Breadth-First Search – explores all neighbors level by level |
| GBFS      | Informed    | Greedy Best-First – uses heuristic to move toward goal |
| A* (AS)   | Informed    | Uses both path cost and heuristic (Euclidean distance) |
| CUS1(IDDFS)      | Uninformed  | Iterative Deepening DFS |
| CUS2(UCS)      | Informed    | Uniform Cost Search – always expands lowest-cost node |

---

## 🗂️ File Structure

```
Assignment_treebased/
├── search.py              # Main Python script containing all algorithms
├── test/                  # Folder containing test cases (e.g., test2.txt)
├── README.md              # This file
├── .gitignore             # Standard Git ignore file
```

---

## 🛠️ How to Run

### ✅ Prerequisites:
- Python 3 installed (tested on Python 3.10+)
- Input `.txt` file structured according to assignment format

### ✅ Command Line Usage:

```bash
python search.py <input_file_path> <method>
```

### ✅ Example:

```bash
python search.py test/test2.txt AS
```

This will run the A* Search on `test2.txt`.

---
