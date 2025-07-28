def find_missing_number(nums):
    i = 0
    n = len(nums)
    while i < n:
        correct_index = nums[i]
        if correct_index < n and nums[i] != nums[correct_index]:
            nums[i],nums[correct_index] = nums[correct_index], nums[i]
        else:
            i+=1
    for i in range(n):
        if nums[i] != i:
            return i
    return n
print(find_missing_number([3, 0, 1]))  # Output: 2