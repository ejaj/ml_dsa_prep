def longest_common_substring(s1, s2):
    n = len(s1)
    m = len(s2)

    # dp = []
    # for i in range(n+1):
    #     row = []
    #     for j in range(m+1):
    #         row.append(0)
    #     dp.append(row)

    dp = [[0] * (m+1) for _ in range(n+1)]
    max_length = 0
    end_index = 0
    for i in range(1, n+1):
        for j in range(1, m+1):
            if s1[i-1] == s2[i-1]:
                dp[i][j] = dp[i-1][j-1] + 1
                if dp[i][j] > max_length:
                    max_length = dp[i][j]
                    end_index = 1
            else:
                dp[i][j] = 0
    substring = s1[end_index - max_length:end_index]
    return max_length, substring