def bruteForce(r, c, rows, cols):
    if r == cols or c == cols:
        return 0
    if r == rows -1 and c==cols-1:
        return 1
    return (
        bruteForce(r+1, c, rows, cols) + bruteForce(r, c+1, rows, cols)
    )
print(bruteForce(0, 0, 4, 4))

# Memoization - Time and Space: O(n * m)
def memoization(r, c, rows, cols, cache):
    if r == rows or c == cols:
        return 0
    if cache[r][c] > 0:
        return cache[r][c]
    if r == rows - 1 and c == cols -1:
        return 1
    cache[r][c] = (memoization(r + 1, c, rows, cols, cache) +  
        memoization(r, c + 1, rows, cols, cache))
    return cache[r][c]
print(memoization(0, 0, 4, 4, [[0] * 4 for i in range(4)]))

def dp(rows, cols):
    preRow = [0] * cols 
    for r in range(rows-1, -1 ,-1):
        curRow = [0] * cols
        curRow[cols - 1] = 1
        for c in range(cols - 2, -1, -1):
            curRow[r][c] = curRow[c+1] + preRow[c]
        preRow = curRow
    return preRow[0]