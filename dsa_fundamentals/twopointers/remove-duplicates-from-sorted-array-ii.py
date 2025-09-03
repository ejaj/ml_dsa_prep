from typing import List
from collections import Counter
class Solution:
    # Brute Force
    # def removeDuplicates(self, nums: List[int]) -> int:
    #     n = len(nums)
    #     if n<=n:
    #         return n
    #     i = 0
    #     while i < n-1:
    #         if nums[i] == nums[i+1]:
    #             j = i +2
    #             cnt = 0
    #             while j < n and nums[i] == nums[j]:
    #                 j+=1
    #                 cnt += 1
    #             for k in range(i+1, n):
    #                 if j >= n:
    #                     break
    #                 nums[k] = nums[j]
    #                 j+=1
    #             n -= cnt
    #             i+2
    #         else:
    #             i += 1
    #     return n
    # Two Pointers
    # def removeDuplicates(self, nums: List[int]) -> int:
    #     l, r = 0, 0
    #     while r < len(nums):
    #         count = 1
    #         while r+1 < len(nums) and nums[r] == nums[r+1]:
    #             r += 1
    #             count += 1
    #         for i in range(min(2, count)):
    #             nums[l] = nums[r]
    #             l+=1
    #         r+=1
    #     return l
    # Hashmap
    def removeDuplicates(self, nums: List[int]) -> int:
        n = len(nums)
        if n <= 2:
            return n
        count = Counter(nums)
        i = 0
        for num in count:
            nums[i] = num
            count[num] -= 1
            i+=1
            if count[num] >= 1:
                nums[i] = num
                count[num] -= 1
                i += 1
        return i
        
              
        
        