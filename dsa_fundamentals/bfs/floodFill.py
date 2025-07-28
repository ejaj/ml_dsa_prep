from collections import deque
def floodFill(image, sr, sc, newColor):
    rows, cols = len(image), len(image[0])
    originalColor =  image[sr][sc]
    if newColor == originalColor:
        return image
    queue = queue()
    queue.append(((sr, sc)))
    image[sr][sc] = newColor
    directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]

    while queue:
         x, y = queue.popleft()
         for dx, dy in directions:
            nx, ny = x + dx, y + dy

            # Check bounds and same original color
            if 0 <= nx < rows and 0 <= ny < cols and image[nx][ny] == originalColor:
                image[nx][ny] = newColor
                queue.append((nx, ny))
    return image

image = [
    [1, 1, 0],
    [1, 0, 0],
    [0, 1, 1]
]

sr, sc = 0, 0  # Starting point
newColor = 2

result = floodFill(image, sr, sc, newColor)

for row in result:
    print(row)