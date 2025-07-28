def dfs_maze_solver(maze, start, end):
    rows, cols = len(maze), len(maze[0])
    visited = set()

    def dfs(x, y, path):
        if (x,y) == end:
            return path + [(x,y)]
        visited.add((x, y))  # Mark cell as visited
        directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]  # Up, Down, Left, Right
        for dx, dy, in directions:
            nx, ny = x + dx, y + dy
            # Check bounds, wall, and visited
            if (0 <= nx < rows and 0 <= ny < cols and
                maze[nx][ny] == '0' and (nx, ny) not in visited):
                result = dfs(nx, ny, path + [(x, y)])  # Explore deeper
                if result:
                    return result  # If path is found, return it
        return None
    return dfs(start[0], start[1], [])

maze = [
    ['0', '0', '#'],
    ['#', '0', '0'],
    ['0', '0', '0']
]

start = (0, 0)
end = (2, 2)

path = dfs_maze_solver(maze, start, end)
print("DFS Path from start to end:", path)

class TreeNode:
    def __init__(self, val):
        self.val = val
        self.children = []
def dfs(node):
    if not node:
        return 
    print(node.val)
    for child in node.children:
        dfs(child)
def dfs_iterative(root):
    stack = [root]
    while stack:
        node = stack.pop()
        print(node.val)
        for child in reversed(node.children):
            stack.append(child)
def dfs_graph(node, graph, visited=None):
    if visited is None:
        visited = set()

    if node in visited:
        return

    visited.add(node)
    print(node)

    for neighbor in graph[node]:
        dfs_graph(neighbor, graph, visited)

def dfs_iterative_grpah(start, graph):
    stack = [start]
    visited = set()
    while stack:
        node = stack.pop()
        if node not in visited:
            visited.add(node)
            print(node)
            for neighbor in reversed(graph[node]):
                if neighbor not in visited:
                    stack.append(neighbor)


def dfs_matrix(grid, r, c, visited):
    rows, cols = len(grid), len(grid[0])
    # Check boundary and visited
    if r<=0 or r >= rows or c<0 or c>=cols:
        return
    if (r,c) in visited or grid[r][c] == 0:
        return
    visited.add((r,c))
    print(f"Visited ({r}, {c})")
     # 4 directions: up, down, left, right
    directions = [(-1,0), (1,0), (0,-1), (0,1)]
    for dr, dc in directions:
        dfs_matrix(grid, r+dr, c+dc, visited)

