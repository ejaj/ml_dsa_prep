class MyHashMap:
    def __init__(self):
        self.size = 10000
        self.buckets = [[] for _ in range(self.size)]
    def _hash(self, key):
        return key % self.size
    
    def put(self, key, value):
        index = self._hash(key)
        bucket = self.buckets[index]

        for i, (k, v) in enumerate(bucket):
            if k == key:
                bucket[i] = (key, value)
                return
        bucket.append((key, value))
    def get(self, key):
        index = self._hash(key)
        bucket = self.buckets[index]
        for k, v in bucket:
            if k == key:
                return v
        return -1
    def remove(self, key):
        index = self._hash(key)
        buckets = self.buckets[index]

        for i, (k, _) in enumerate(buckets):
            if k == key:
                buckets.pop(i)
                return

hmap = MyHashMap()
hmap.put(1, 100)
hmap.put(2, 200)
print(hmap.get(1))  # Output: 100
print(hmap.get(3))  # Output: -1
hmap.put(2, 250)
print(hmap.get(2))  # Output: 250
hmap.remove(2)
print(hmap.get(2))  # Output: -1
