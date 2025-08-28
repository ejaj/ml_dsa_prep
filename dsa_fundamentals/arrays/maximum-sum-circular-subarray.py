from typing import List

class Solution:
    def maxSubarraySumCircular(self, nums: List[int]) -> int:
        gloal_max, global_min = nums[0], nums[0]
        cur_max, cur_min = 0, 0
        total = 0
        for num in nums:
            cur_max = max(cur_max + num, num)
            cur_min = min(cur_min + num, num)
            total += num 
            gloal_max = max(gloal_max, cur_max)
            global_min = min(global_min, cur_min)
        return max(gloal_max, total - global_min) if gloal_max > 0 else gloal_max
    def max_sub_cir_br(self, nums:List[int]) -> int:
        n = len(nums)
        res = nums[0]

        for i in range(n):
            cur_sum = 0
            for j in range(i, i + n):
                cur_sum += nums[j % n]
                res = max(res, cur_sum)

        return res