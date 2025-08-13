from collections import deque
from typing import Optional


# Definition for a Node.
class Node:
    def __init__(self, val=0, neighbors=None):
        self.val = val
        self.neighbors = neighbors if neighbors is not None else []


class Solution:
    def cloneGraph(self, node: Optional['Node']) -> Optional['Node']:
        old_to_new = {}

        def dfs(node):
            if node in old_to_new:
                return old_to_new[node]
            copy = Node(node.val)
            old_to_new[node] = copy
            for neighbor in node.neighbors:
                copy.neighbors.append(dfs(neighbor))
            return copy

        return dfs(node) if node else None

    def cloneGraphBFS(self, node: Optional['Node']) -> Optional['Node']:
        if not node:
            return None
        old_to_new = {}
        old_to_new[node] = Node(node.val)
        q = deque([node])
        while q:
            current = q.popleft()
            for neighbor in current.neighbors:
                if neighbor not in old_to_new:
                    old_to_new[neighbor] = Node(neighbor.val)
                    q.append(neighbor)
                old_to_new[current].neighbors.append(old_to_new[neighbor])
        return old_to_new[node]
