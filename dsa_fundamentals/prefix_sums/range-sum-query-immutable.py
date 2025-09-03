from typing import List

# Brute Force
# class NumArray:
#     def __init__(self, nums: List[int]):
#         self.nums = nums

#     def sumRange(self, left: int, right: int) -> int:
#         res = 0
#         for i in range(left, right+1):
#             res += self.nums[i]
#         return res

# Prefix Sum - I
# class NumArray:
#     def __init__(self, nums: List[int]):
#         self.prefix = []
#         cur = 0
#         for num in nums:
#             cur += num
#             self.prefix.append(cur)

#     def sumRange(self, left: int, right: int) -> int:
#         right_sum = self.prefix[right]
#         left_sum = self.prefix[left - 1] if left > 0 else 0
#         return right_sum - left_sum
# Prefix Sum - II
class NumArray:
    def __init__(self, nums: List[int]):
        self.prefix = [0] * (len(nums) + 1)
        for i in range(len(nums)):
            self.prefix[i + 1] = self.prefix[i] + nums[i]

    def sumRange(self, left: int, right: int) -> int:
        return self.prefix[right+1] - self.prefix[left]