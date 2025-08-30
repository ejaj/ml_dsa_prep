from typing import List

class Solution:
    # Brute Force
    # def containsNearbyDuplicate(self, nums: List[int], k: int) -> bool:
    #     for L in range(len(nums)):
    #         for R in range(L + 1, min(len(nums), L + k + 1)):
    #             if nums[L] == nums[R]:
    #                 return True
    #     return False
    # Hash MAp
    # def containsNearbyDuplicate(self, nums: List[int], k: int) -> bool:
    #     mp = {}
    #     for i in range(len(nums)):
    #         if nums[i] in mp and i - mp[nums[i]] <= k:
    #             return True
    #         mp[nums[i]] = i
    # hash Set
    def containsNearbyDuplicate(self, nums: List[int], k: int) -> bool:
        window = set()
        L = 0

        for R in range(len(nums)):
            if R - L > k:
                window.remove(nums[L])
                L += 1
            if nums[R] in window:
                return True
            window.add(nums[R])

        return False