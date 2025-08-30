from typing import List

class Solution:
    # Brute Force
    # def minSubArrayLen(self, target: int, nums: List[int]) -> int:
    #     n = len(nums)
    #     res = float("inf")
    #     for i in range(n):
    #         cur_sum = 0
    #         for j in range(i, n):
    #             cur_sum += nums[j]
    #             if cur_sum >= target:
    #                 res = min(res, j-i+1)
    #                 break

    #     return 0 if  res == float("inf") else res
    # Sliding window
    def minSubArrayLen(self, target: int, nums: List[int]) -> int:
        l, total = 0, 0
        res = float("inf")
        for r in range(len(nums)):
            total += nums[r]
            while total >= target:
                res = min(r-l+1, res)
                total -= nums[l]
                l +=1
        return 0 if res == float("inf") else res
