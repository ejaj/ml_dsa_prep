from typing import List

class Solution:
    # Brute Force
    # def maxArea(self, height: List[int]) -> int:
    #     res = 0
    #     for i in range(len(height)):
    #         for j in range(i+1, len(height)):
    #             res = max(res, min(height[i], height[j]) * (j - i))
    #     return res
    # Tow pointers
    def maxArea(self, height: List[int]) -> int:
        l, r = 0, len(height) - 1
        res = 0
        while l < r:
            area = min(height[l], height[r]) * (r-l)
            res = max(res, area)
            if height[l] <= height[r]:
                l += 1
            else:
                r -= 1
        return res
