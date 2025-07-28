def count_bits(n):
    dp = [0] * (n+1)
    for i in range(1, n+1):
        dp[i] = dp[i>>1] + dp[i&1] # i // 2 + (1 if i is odd)
    return dp