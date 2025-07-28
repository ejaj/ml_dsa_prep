nums = [1, 2, 4, 7, 11, 15]
target = 15


# time = 0(n**2)
# brut force
def two_sum(nums):
    for i in range(len(nums)):
        for j in range(i+1, len(nums)):
            if nums[i] + nums[i+1] == target:
                return True
    return False

# Two pointer, two pointer only works sorted array

def two_sum_pointer(nums):
    left = 0
    right = len(nums) - 1

    while left < right:
        total = nums[left] + nums[right]
        if total == target:
            print(f"Found: {nums[left]} + {nums[right]} = {target}")
            break
        elif total < target:
            left += 1 # move right to get bigger number
        else:
            right += 1 #  # move left to get smaller number

def tow_sum_without_sorted_arr(nums, target):
    seen = set()
    for num in nums:
        needed = target - num
        if needed in seen:
            return True, (needed, num)
        seen.add(num)
    return False, ()
nums = [4, 1, 7, 11, 2, 15]
target = 15

found, pair = tow_sum_without_sorted_arr(nums, target)
if found:
    print("Pair found:", pair)
else:
    print("No pair found")


def count_frequnices(nums):
    freq = {}
    for i, num in enumerate(nums):
        if num in freq:
            freq[num] += 1
        else:
            freq[num] = 1
    return freq
        
def length_of_longest_substring(s):
    seen = set()
    left = 0
    max_len = 0

    for right in range(len(s)):
        while s[right] in seen:
            seen.remove(s[left])
            left += 1
        seen.add(s[right])
        max_len = max(max_len, right - left + 1)

    return max_len

def is_valid(s):
    stack = []
    mapping = {')': '(', ']': '[', '}': '{'}
    for ch in s:
        if ch in mapping.values(): # If it's an opening bracket
            stack.append(ch)
        elif ch in mapping: # If it's a closing bracket
            if not stack or stack[-1] == mapping[ch]:
                return False
            stack.pop()
        else:
            return False

class Stack:
    def __init__(self):
        self.stack = []
    def push(self, x):
        self.stack.append(x)
    def pop(self):
        if self.stack:
            self.stack.pop()
    def top(self):
        if self.stack:
            return self.stack[-1]
        return None
    def get_min(self):
        if not self.stack:
            return None
        minimum = self.stack[0]
        for num in self.stack[1:]:
            if num < minimum:
                minimum = num
        return minimum
    
class MinStack:
    def __init__(self):
        self.stack = []
        self.min_stack = []
    def push(self, x):
        self.stack.append(x)
        # Push to min_stack only if it's the first element or x <= current min

        if not self.min_stack or x <= self.min_stack[-1]:
            self.min_stack.append(x)
    def pop(self):
        popped = self.stack.pop()
        if popped == self.min_stack[-1]:
            self.min_stack.pop() 
    def top(self):
        return self.stack[-1] if self.stack else None
    def get_min(self):
        return self.min_stack[-1] if self.min_stack else None

def merge_sorted_arrays(nums1, nums2):
    p1, p2 = 0, 0
    resuelt = []
    while p1 < len(nums1) and p2 < len(nums2):
        if nums1[p1] < nums2[p2]:
            resuelt.append(nums1[p1])
            p1+=1
        else:
            resuelt.append(nums2[p2])
            p2 += 1
    
    # Add reming elements
    resuelt.extend(nums1[p1:])
    resuelt.extend(nums2[p2:])
    return resuelt

def remove_duplicates(nums):
    if not nums:
        return 0
    slow = 0
    for fast in range(1, len(nums)):
        if nums[fast] != nums[slow]:
            slow += 1
            nums[slow] = nums[fast]
    return slow+1
nums = [1, 1, 2, 2, 3]
length = remove_duplicates(nums)
print(nums[:length])  # Output: [1, 2, 3]