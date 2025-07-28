class Array:
    def __init__(self):
        self.capacity = 2
        self.length = 0
        self.arr = [0] * self.capacity

    def resize(self):
        # Create new array of double capacity
        self.capacity = 2 * self.capacity
        new_arr = [0] * self.capacity

        for i in range(self.length):
            new_arr[i] = self.arr[i]
        self.arr = new_arr

    def pushback(self, n):
        if self.length == self.capacity:
            self.resize()
        self.arr[self.length] = n
        self.length += 1

    def popback(self):
        if self.length > 0:
            self.length -= 1

    def get(self, index):
        if index < self.length:
            return self.arr[index]

    def insert(self, index, value):
        if index < self.length:
            self.arr[index] = value
            return

    def print_array(self):
        for i in range(self.length):
            print(self.arr[i])
        print()


arr = Array()

# Inserting elements
arr.insert(0, 10)  # Inserts at index 0
arr.insert(1, 20)  # Inserts at index 1
arr.insert(2, 30)  # Requires resizing
arr.insert(3, 40)

print("Array after inserting elements:")
arr.print_array()

# Access elements
print("Element at index 2:", arr.get(2))

# Modify element
arr.insert(2, 99)
print("Array after modifying index 2:")
arr.print_array()

arr.popback()
print("Array after popback:")
arr.print_array()
