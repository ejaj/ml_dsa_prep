def subsets(nums):
    result = []
    def backtrack(index, path):
        result.append(path)
        for i in range(index, len(nums)):
            backtrack(i+1, path + [nums[i]])
    backtrack(0, [])
    return result
def subsets_bitmask(nums):
    n = len(nums)
    result = []
    for i in range(2**n):
        subsets = []
        for j in range(n):
            if i &(1<<j):
                subsets.append(nums[j])
        result.append(subsets)
    return result