# Definition for singly-linked list.
class ListNode:
    def __init__(self, val=0, next=None):
        self.val = val
        self.next = next


class Solution:
    def reverseList(self, head: ListNode) -> ListNode:
        prev, curr = None, head

        while curr:
            temp = curr.next
            curr.next = prev
            prev = curr
            curr = temp
        return prev
    
    def reverseListRecursive(self, head: ListNode) -> ListNode:
        if not head:
            return None
        
        new_head = head
        if head.next:
            new_head = self.reverseListRecursive(head.next)
            head.next.next = head
        head.next =  None
        return new_head
