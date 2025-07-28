class ListNode:
    def __init__(self, val):
        self.val = val
        self.next = None
        self.prev = None
    

class LinkedList:
    def __init__(self):
        self.head = ListNode(-1)
        self.tail = ListNode(-1)
        self.head.next = self.tail
        self.tail.prev = self.head
    def insert_front(self, val):
        new_node = ListNode(val)
        new_node.prev =self.head
        new_node.next = self.head.next

        self.head.next.prev = new_node
        self.head.next = new_node

    def insert_end(self, val):
        new_node = ListNode(val)

        new_node.next = self.tail
        new_node.prev = self.tail.prev

        self.tail.prev.next = new_node
        self.tail.prev = new_node

    def remove_front(self):
        self.head.next.next.prev = self.head
        self.head.next = self.head.next.next
    
    def remove_end(self):
        self.tail.prev.prev.next = self.tail
        self.tail.prev = self.tail.prev.prev

    def print(self):
        curr = self.head.next
        while curr != self.tail:
            print(curr.val, " -> ")
            curr = curr.next
        print()
    

class DBlyLinkedList:
    def __init__(self):
        self.head = None
    
    def append(self, data):
        new_node = ListNode(data)

        if self.head is None:
            self.head = new_node
            return
        temp = self.head
        while temp.next:
            temp = temp.next
        temp.next = new_node
        new_node.prev = temp
    
    def print_forward(self):
        temp = self.head
        while temp:
            print(temp.data, end="")
            temp = temp.next
        print()
    def print_backward(self):
        temp = self.head
        if not temp:
            return
        while temp.next:
            temp = temp.next
        while temp:
            print(temp.data, end='')
            temp = temp.prev
        print()