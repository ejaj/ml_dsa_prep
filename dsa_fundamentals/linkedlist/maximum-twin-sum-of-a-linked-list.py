from typing import Optional

# Definition for singly-linked list.
class ListNode:
    def __init__(self, val=0, next=None):
        self.val = val
        self.next = next

class Solution:
    # Convert to Array
    # def pairSum(self, head: Optional[ListNode]) -> int:
    #     arr = []
    #     cur = head 
    #     while cur:
    #         arr.append(cur.val)
    #         cur = cur.next 
    #     i, j = 0, len(arr) - 1
    #     res = 0
    #     while i<j:
    #         res = max(res, arr[i] + arr[j])
    #         i, j = i + 1, j-1
    #     return res 
    # reverse the first half
    def pairSum(self, head: Optional[ListNode]) -> int:
        slow, fast = head, head 
        prev = None 
        while fast and fast.next:
            fast = fast.next.next 
            tmp = slow.next 
            slow.next = prev 
            prev = slow 
            slow = tmp  
        res = 0 
        while slow:
            res = max(res, prev.val + slow.val)
            prev = prev.next 
            slow = slow.next 
        return res