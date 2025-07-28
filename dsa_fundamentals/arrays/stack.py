class Stack:
    def __init__(self):
        self.stack = []  # Initialize an empty list to act as a stack

    def push(self, n):
        """Push an element onto the stack."""
        self.stack.append(n)

    def pop(self):
        """Remove and return the top element from the stack."""
        if not self.is_empty():
            return self.stack.pop()
        return "Stack is empty"

    def peek(self):
        """Return the top element without removing it."""
        if not self.is_empty():
            return self.stack[-1]
        return "Stack is empty"

    def is_empty(self):
        """Check if the stack is empty."""
        return len(self.stack) == 0

    def size(self):
        """Return the number of elements in the stack."""
        return len(self.stack)

    def print_stack(self):
        """Print the stack elements."""
        print(self.stack)


s = Stack()

# Pushing elements onto the stack
s.push(10)
s.push(20)
s.push(30)

print("Stack after pushing elements:")
s.print_stack()  # Output: [10, 20, 30]

# Getting the top element
print("Top element:", s.peek())  # Output: 30

# Popping an element
print("Popped:", s.pop())  # Output: 30

print("Stack after popping:")
s.print_stack()  # Output: [10, 20]

# Checking stack size
print("Stack size:", s.size())  # Output: 2

# Checking if stack is empty
print("Is stack empty?", s.is_empty())  # Output: False

# Popping remaining elements
s.pop()
s.pop()
print("Is stack empty after popping all elements?", s.is_empty())  # Output: True

# Trying to pop from an empty stack
print("Popped:", s.pop())  # Output: Stack is empty
