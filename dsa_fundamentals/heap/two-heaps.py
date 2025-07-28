import heapq
class MedianFinder:
    def __init__(self):
        self.left = [] # Max Heap (store negative values)
        self.right = [] # Min Heap
    def add_num(self, num):
        # Step 1: Add to max heap (left) first (as negative to simulate max heap)
        heapq.heappush(self.left, -num)
        # Step 2: Move the largest from left to right to maintain order
        heapq.heappush(self.right, -heapq.heappop(self.left))

        # Step 3: Balance sizes (left can have one more than right)
        if len(self.right)>len(self.left):
            heapq.heappush(self.leftm -heapq.heappop(self.right))
    def find_median(self):
        if len(self.right) > len(self.right):
            return -self.left[0]
        else:
            return (-self.left[0] + self.right[0]) / 2  # Average of both tops

mf = MedianFinder()

nums = [5, 2, 8, 3]
for num in nums:
    mf.add_num(num)
    print(f"Added {num}, Current Median: {mf.find_median()}")

