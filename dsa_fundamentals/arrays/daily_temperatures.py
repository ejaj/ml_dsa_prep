def daily_temperatures(temperatures):
    result = [0] * len(temperatures)
    stack = []

    for i, temp in enumerate(temperatures):
        while stack and temperatures[stack[-1]] < temp:
            pre_day = stack.pop()
            result[pre_day] = i - pre_day
        stack.append(i)
    return result

def subarray_sum(nums, k):
    count = 0
    for start in range(len(nums)):
        sum_ = 0
        for end in range(start, len(nums)):
            sum_ += nums[end]
            if sum_ == k:
                count +=1
    return count
from collections import defaultdict
def subarray_sum_hasmap(nums, k):
    count = 0
    curr_sum = 0
    prefix_sums = defaultdict(int)
    prefix_sums[0] = 1
    for num in nums:
        curr_sum += num
        if (curr_sum - k) in prefix_sums:
            count += prefix_sums[curr_sum - k]
        prefix_sums[curr_sum] += 1
    return count


def length_of_longest_substring(s: str) -> int:
    char_index = {}
    max_length = start = 0
    for end, char in enumerate(s):
        if char in char_index and char_index[char] >= start:
            start = char_index[char] + 1
        char_index[char] = end
        max_length = max(max_length, end - start +1)
    return max_length

def anagram_map(strs):
    anagram_map = defaultdict(list)
    for word in strs:
        sorted_word = ''.join(sorted(word))
        anagram_map[sorted_word].append(word)
    return list(anagram_map.values())

def three_sum_brute_force(nums):
    n = len(nums)
    reseult = set()
    for i in range(n):
        for j in range(i+1, n):
            for k in range(j+1, n):
                if nums[i] + nums[j] + nums[j] == 0:
                    triplet = tuple(sorted([nums[i], nums[j], nums[k]]))
                    reseult.append(triplet)
    return [list(triplet) for triplet in reseult]

def three_sum_opti(nums):
    nums.sort()
    result = []
    for i in range(len(nums)):
        if i >0 and nums[i] == nums[i-1]:
            continue

        left, right = i+1, len(nums) - 1
        while left < right:
            total = nums[i] + nums[left] + nums[right]
            if total == 0:
                result.append([nums[i], nums[left], nums[right]])
                while left < right and nums[left] == nums[left+1]:
                    left += 1
                while left < right and nums[right] == nums[right-1]:
                    right -= 1
                left +=1
                right -= 1

            elif total < 0:
                left += 1
            else:
                right -= 1
    return result