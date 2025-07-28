def permute(nums):
    result = []
    def backtrack(path):
        # If we picked all numbers, save the current arrangement
        if len(path) == len(nums):
            result.append(path[:])
            return
        # Try every number in nums
        for num in nums:
            if num in path:
                continue
            path.append(num)
            backtrack(path)
            path.pop()
    backtrack([])

    return result
nums = [1, 2]
permute(nums)
