class ListNode:
    def __init__(self, val):
        self.val = val
        self.next = None
    
class LinkedList:
    def __init__(self):
        self.head = ListNode(-1)
        self.tail = self.head
    
    def insert_end(self, val):
        self.tail.next = ListNode(val)
        self.tail = self.tail.next
    
    def remove(self, index):
        if not self.head:
            return

        if index == 0:
            self.head = self.head.next
            if self.head is None:
                self.tail = None
            return
        i = 0
        curr = self.head
        while i < index-1 and curr.next:
            i+=1
            curr = curr.next
        if curr.next:
            if curr.next == self.tail:
                self.tail = curr
            curr.next = curr.next.next
    def print(self):
        curr = self.head.next
        while curr:
            print(curr.valu, " -> ", end="")
            curr=curr.next
        print()


