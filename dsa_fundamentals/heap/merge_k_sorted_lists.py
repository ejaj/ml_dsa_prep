import heapq

def merge_k_sorted_lists(lists):
    heap = []
    for i in range(len(lists)):
        if lists[i]:
            heapq.heappush(heap, (lists[i][0], i, 0))
    result = []
    while heap:
        val, list_idx, elem_idx = heapq.heappop(heap)
        result.append(val)
        if elem_idx + 1 < len(lists[list_idx]):
            next_val = lists[list_idx][elem_idx + 1]
            heapq.heappush(heap, (next_val, list_idx, elem_idx+1))
    return result
