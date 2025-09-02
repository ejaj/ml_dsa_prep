from typing import List
from collections import defaultdict

class Solution:
    # Brute Force
    # def twoSum(self, numbers: List[int], target: int) -> List[int]:
    #     for i in range(len(numbers)):
    #         for j in range(i+1, len(numbers)):
    #            if numbers[i] + numbers[j] == target:
    #                return[i+1, j+1]
    #     return []
    # Binary search
    # def twoSum(self, numbers: List[int], target: int) -> List[int]:
    #     for i in range(len(numbers)):
    #         l, r = i+1, len(numbers) -1
    #         temp = target - numbers[i]
    #         while l<r:
    #             mid = l + (r-l)// 2
    #             if numbers[mid] == temp:
    #                 return[i+1, mid+1]
    #             elif numbers[mid] < temp:
    #                 l = mid+1
    #             else:
    #                 r = mid- 1
    #     return []
    # Hash Map
    # def twoSum(self, numbers: List[int], target: int) -> List[int]:
    #     mp = defaultdict(int)
    #     for i in range(len(numbers)):
    #         temp = target - numbers[i]
    #         if mp[temp]:
    #             return[mp[temp], i+1]
    #         mp[numbers[i]] = i+1
    # Tow pointer
    def twoSum(self, numbers: List[int], target: int) -> List[int]:
        l, r = 0, len(numbers) - 1
        while l<r:
            cur_sum = numbers[l] + numbers[r]
            if cur_sum > target:
                r -= 1
            elif cur_sum < target:
                l+=1
            else:
                return[l+1, r+1]
        return []
        