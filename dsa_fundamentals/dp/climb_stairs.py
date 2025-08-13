def climb_stairs(n):
    if n == 0 or n ==1:
        return 1
    dp = [0] * (n + 1)  # Create a list to store answers
    dp[0] = 1
    dp[1] = 1
    for i in range(2, n+1):
        dp[i] = dp[i-1] + dp[i-2]
    print(dp[n])
climb_stairs(6)

def minCostClimbingStairs(cost):
    n = len(cost)
    dp = [0] * n
    dp[0] = cost[0]
    dp[1] = cost[1]

    for i in range(2, n):
        dp[i] = cost[i] + min(dp[i-1], dp[i-2])
    return min(dp(n-1), dp[n-2])

def climb_stairs(n):
    one, two = 1, 1
    for i in range(n-1):
        temp = one
        one = one+two
        two = temp
    return one