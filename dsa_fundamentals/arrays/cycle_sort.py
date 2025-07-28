def cyclic_sort(nums):
    i = 0
    while i < len(nums):
        current_index = nums[i]
        if nums[i] != nums[current_index]:
            nums[i], nums[current_index] = nums[current_index], nums[i]
        else:
            i+=1
    return nums

nums = [3, 1, 2, 0]
print(cyclic_sort(nums))  # Output: [0, 1, 2, 3]