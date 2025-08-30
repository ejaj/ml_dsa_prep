from typing import List

class Solution:
    # Brute Force
    # def numOfSubarrays(self, arr: List[int], k: int, threshold: int) -> int:
    #     res = 0
    #     l = 0
    #     for r in range(k-1, len(arr)):
    #         sum_ = 0
    #         for i in range(l, r+1):
    #             sum_ += arr[i]
    #         if sum_ / k >= threshold:
    #             res += 1
    #         l += 1
    #     return res
    # Sliding window
    def numOfSubarrays(self, arr: List[int], k: int, threshold: int) -> int:
        res = 0
        window_sum = sum(arr[:k])
        if window_sum / k >= threshold:
            res += 1
        for i in range(k, len(arr)):
            window_sum += arr[i] - arr[i-k]
            if window_sum / k >= threshold:
                res += 1
        return res

    