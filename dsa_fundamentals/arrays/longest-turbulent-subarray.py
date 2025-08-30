from typing import List

class Solution:

    # Sliding window
    def maxTurbulenceSize(self, arr: List[int]) -> int:
        l, r, res, prev = 0, 1, 1, ""

        while r < len(arr):
            if arr[r-1] > arr[r] and prev != ">":
                res = max(res, r-l+1)
                r += 1
                prev = ">"
            elif arr[r-1] < arr[r] and prev != "<":
                res = max(res, r-l+1)
                r += 1
                prev = "<"
            else:
                r = r+1 if arr[r] == arr[r-1] else r 
                l = r -1
                prev = ""
        return res 


    # Brute Force
    # def maxTurbulenceSize(self, arr: List[int]) -> int:
    #     n = len(arr)
    #     res = 1
    #     for i in range(n-1):
    #         if arr[i] == arr[i+1]:
    #             continue
    #         sign = 1 if arr[i] > arr[i+1] else 0
    #         j = j+1
    #         while j < n-1:
    #             if arr[j] == arr[j+1]:
    #                 break
    #             cur_sign = 1 if arr[j] > arr[j+1] else 0
    #             if sign == cur_sign:
    #                 break

    #             sign = cur_sign
    #             j+=1
    #         res = max(res, j-i+1)
    #     return res