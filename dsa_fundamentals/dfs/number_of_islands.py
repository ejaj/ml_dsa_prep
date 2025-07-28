from collections import deque
def numIslands_dfs(grid):
    if not grid:
        return 0
    
    rows, cols = len(grid), len(grid[0])
    count = 0

    # Define DFS function to flood connected land
    def dfs(i,j):
        # If out of bounds or water, stop
        if i < 0 or i >= rows or j < 0 or j >= cols or grid[i][j] != '1':
            return
        # Mark this cell as visited (turn it into water)
        grid[i][j] = '0'
        dfs(i + 1, j)  # down
        dfs(i - 1, j)  # up
        dfs(i, j+1) # right
        dfs(i, j-1) # left
    # Main loop to find islands
    for i in range(rows):
        for j in range(cols):
            if grid[i][j] == '1':
                dfs(i,j)
                count += 1
    return count

def num_island_dfs(grid):
    if not grid:
        return 0
    
    rows, cols = len(grid), len(grid[0])
    count = 0

    def dfs(i, j):
        if i < 0 or i >= rows or j<0 or j >= cols or grid[i][j] != '1':
            return
        grid[i][j] = '0'
        dfs(i+1, j) # down
        dfs(i-1, j) # uo
        dfs(i, j+1) # rigth
        dfs(i, j-1) # left
    for i in range(rows):
        for j in range(cols):
            if grid[i][j] == '1':
                dfs(i,j)
                count += 1
    return count


def numIslands_bfs(grid):
    if not grid:
        return 0
    rows, cols = len(grid), len(grid[0])
    count = 0
     # Directions: up, down, left, right
    directions = [(-1,0), (1,0), (0,-1), (0,1)]

    def bfs(i,j):
        queue = deque()
        queue.append(i,j)
        grid[i][j] = '0'  # mark as visited
        while queue:
            x, y = queue.popleft()
            for dx, dy in directions:
                nx, ny = x+dx, y+dy
                if 0 <= nx < rows and 0 <= ny < cols and grid[nx][ny] == '1':
                    queue.append((nx, ny))
                    grid[nx][ny] = '0'  # mark as visited


    for i in range(rows):
        for j in range(cols):
            if grid[i][j] == '1':
                bfs(i, j)
                count += 1


grid = [
    ['1', '1', '0', '0'],
    ['0', '1', '0', '0'],
    ['0', '0', '1', '0'],
    ['1', '0', '0', '1']
]

print(numIslands_dfs(grid))  # Output: 4