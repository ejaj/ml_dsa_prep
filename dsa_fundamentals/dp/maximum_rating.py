def maximizeRatings(ratings):
    from functools import lru_cache
    n = len(ratings)
    @lru_cache(None)
    def dp(i, skpped_prev):
        if i>= n:
            return 0
        take = ratings[i] + dp(i+1, False)
        skip = float('-inf')
        if not skpped_prev:
            skip = dp(i+1, True)
        return max(take, skip)
    return dp(0, False)
