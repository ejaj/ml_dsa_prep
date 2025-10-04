# Definition for singly-linked list.
from typing import Optional

class ListNode:
    def __init__(self, val=0, next=None):
        self.val = val
        self.next = next
class Solution:
    # Convert To Array
    # def middleNode(self, head: Optional[ListNode]) -> Optional[ListNode]:
    #     cur = head 
    #     arr = []
    #     while cur: 
    #         arr.append(cur)
    #         cur = cur.next 
    #     return arr[len(arr) // 2]
    def middleNode(self, head: Optional[ListNode]) -> Optional[ListNode]:
        slow, fast = head, head

        while fast and fast.next:
            slow = slow.next
            fast = fast.next.next
        return slow