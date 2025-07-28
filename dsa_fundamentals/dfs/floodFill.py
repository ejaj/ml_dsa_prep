def floodFill(image, sr, sc, newColor):
    rows, cols = len(image), len(image[0])
    originalColor = image[sr][sc]
    if originalColor == newColor:
        return image
    # DFS function to paint all connected same-color cells
    def dfs(i, j):
        # If out of bounds or not the original color, stop
        if i < 0 or i >= rows or j < 0 or j >= cols or image[i][j] != originalColor:
            return
        image[i][j] = newColor
        # Recursively visit all 4 neighbors
        dfs(i + 1, j)  # down
        dfs(i - 1, j)  # up
        dfs(i, j + 1)  # right
        dfs(i, j - 1)  # left

    dfs(sr,sc)
    return image
image = [
    [1, 1, 0],
    [1, 0, 0],
    [0, 1, 1]
]

start_row = 0
start_col = 0
new_color = 2

result = floodFill(image, start_row, start_col, new_color)

# Print the result
for row in result:
    print(row)