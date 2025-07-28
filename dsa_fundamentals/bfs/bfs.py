from collections import deque

def bfs_maze():
    maze = [
        ['S', '.', '.'],
        ['#', '#', '.'],
        ['.', '.', 'E']
    ]
    rows, cols = len(maze), len(maze[0])
    start = (0,0)
    queue = deque()
    queue.append((start, [start]))
    # print(queue)
    visited = set()
    visited.add(start)
    
    directions = [(-1,0), (1,0), (0,-1), (0,1)]  # up, down, left, right

    while queue:
        (x,y), path = queue.popleft()
        if maze[x][y] == 'E':
            return path
        for dx, dy in directions:
            nx, ny = x+dx, y+dy
            if 0 <= nx < rows and 0 <= ny < cols:
                queue.append((nx, ny), path+[(nx, ny)]
                )
                visited.add((nx, ny))

    return None

bfs_maze()

def bfs_graph(graph, start):
    visited = set()
    queue = deque([start])
    order = []
    while queue:
        node = queue.popleft()
        if node not in visited:
            visited.add(node)
            order.append(node)
            print(f"Visited {node}")

            for neighbor in graph[node]:
                if neighbor not in visited:
                    queue.append(neighbor)
    return order
graph = {
    'A': ['B', 'C'],
    'B': ['D', 'E'],
    'C': ['F'],
    'D': [],
    'E': ['F'],
    'F': []
}
result = bfs_graph(graph, 'A')
print("\nBFS Traversal Order:", result)
class TreeNode:
    def __init__(self, val):
        self.val = val
        self.children = []
def bfs(root):
    queue = deque([root])
    while queue:
        node = queue.popleft()
        print(node.val)
        for child in node.children:
            queue.append(child)

def bfs_graph_b(start, graph):
    visited = set()
    queue = deque[start]
    visited.add(start)
    while True:
        node = queue.popleft()
        print(node)
        for neighbor in graph[node]:
            if neighbor not in visited:
                visited.add(neighbor)
                queue.append(neighbor)

def bfs_matrix(grid, start):
    rows, cols = len(grid), len(grid[0])
    visited = set()
    queue = deque([start])
    visited.add(start)
    directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]  # up, down, left, right
    while queue:
        r, c = queue.popleft()
        print(f"Visited: ({r}, {c}) = {grid[r][c]}")
        for dx, dy in directions:
            nr, nc = r+dx, c+dy
            # Check boundaries and not visited
            if 0 <= nr < rows and 0 <= nc < cols and (nr, nc) not in visited:
                visited.add((nr, nc))
                queue.append((nr, nc))

grid = [
  [1, 1],
  [0, 1]
]
bfs_matrix(grid, (0, 0))