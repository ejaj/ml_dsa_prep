import heapq
from collections import Counter

def top_k_frequent(nums, k):
    freq_map = Counter(nums)
    # print(freq_map)
    min_q = []
    for num, freq in freq_map.items():
        heapq.heappush(min_q, (freq, num))
        if len(min_q) > k:
            heapq.heappop(min_q) 
    # print(min_q)
    # print(len(min_q))
    return [num for freq, num in min_q]


arr = [1, 1, 1, 2, 2, 3]
k = 2
print(top_k_frequent(arr, k))