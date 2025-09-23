from typing import List


class Solution:
    
    # Brute Force
    # def pivotIndex(self, nums: List[int]) -> int:
    #     n = len(nums)
    #     for i in range(n):
    #         leftsum = rightsum = 0
    #         for l in range(i):
    #             leftsum += nums[l]
    #         for r in range(i+1, n):
    #             rightsum += nums[r]
    #         if leftsum == rightsum:
    #             return i
    #     return -1
    # Prefix Sum
    # def pivotIndex(self, nums: List[int]) -> int:
    #     n = len(nums)
    #     prefix_sum = [0] * (n+1)
    #     for i in range(n):
    #         prefix_sum[i+1] = prefix_sum[i] + nums[1]
    #     for i in range(n):
    #         left_sum = prefix_sum[i]
    #         right_sum = prefix_sum[n] - prefix_sum[i+1]
    #         if left_sum == right_sum:
    #             return i
    #     return -1
    # 3. Prefix Sum (Optimal)
    def pivotIndex(self, nums: List[int]) -> int:
        total = sum(nums)
        leftsum = 0
        for i in range(len(nums)):
            rightsum = total - nums[i] - leftsum
            if leftsum == rightsum:
                return i
            leftsum += nums[i]
        return -1


        