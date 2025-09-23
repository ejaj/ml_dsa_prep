from typing import List

class Solution:
    # Brute Force
    # def subarraySum(self, nums: List[int], k: int) -> int:
    #     res = 0
    #     for i in range(len(nums)):
    #         sum = 0
    #         for j in range(i, len(nums)):
    #             sum += nums[j]
    #             if sum == k:
    #                 res += 1
    #     return res
     def subarraySum(self, nums: List[int], k: int) -> int:
        res = cursum = 0
        prefix_sum = {0:1}
        for num in nums:
            cursum += num
            diff = cursum - k 
            res += prefix_sum.get(diff, 0)
            prefix_sum[cursum] = 1 + prefix_sum.get(cursum, 0)
        return res