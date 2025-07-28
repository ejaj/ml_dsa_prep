from typing import List, Optional

class TreeNode:
    def __init__(self, key: int, val: int):
        self.key = key
        self.val = val
        self.left: Optional['TreeNode'] = None
        self.right: Optional['TreeNode'] = None

class TreeMap:
    
    def __init__(self):
        self.root: Optional[TreeNode] = None        


    def insert(self, key: int, val: int) -> None:
        def insert_node(node: Optional[TreeNode], key: int, val: int) -> TreeNode:
            if not node:
                return TreeNode(key, val)
            if key < node.key:
                node.left = insert_node(node.left, key, val)
            elif key > node.key:
                node.right = insert_node(node.right, key, val)
            else:
                node.val = val
            return node
        self.root = insert_node(self.root, key, val)
    
    def get(self, key: int) -> int:
        node = self.root
        while node:
            if key < node.key:
                node = node.left
            elif key > node.key:
                node = node.right
            else:
                return node.val
        return -1


    def getMin(self) -> int:
        if not self.root:
            return -1
        node = self.root
        while node.left:
            node = node.left
        return node.val


    def getMax(self) -> int:
        if not self.root:
            return -1
        node = self.root
        while node.right:
            node = node.right
        return node.val


    def remove(self, key: int) -> None:
        def delete_node(node: Optional[TreeNode], key: int) -> Optional[TreeNode]:
            if not node:
                return None
            if key < node.key:
                node.left = delete_node(node.left, key)
            elif key > node.key:
                node.right = delete_node(node.right, key)
            else:
                if not node.left:
                    return node.right
                if not node.right:
                    return node.left
                # Get the smallest node in the right subtree
                successor = node.right
                while successor.left:
                    successor = successor.left
                node.key, node.val = successor.key, successor.val
                node.right = delete_node(node.right, successor.key)
            return node
        self.root = delete_node(self.root, key)



    def getInorderKeys(self) -> List[int]:
        result = []
        def inorder(node: Optional[TreeNode]):
            if node:
                inorder(node.left)
                result.append(node.key)
                inorder(node.right)
        inorder(self.root)
        return result

t = TreeMap()
t.insert(1, 2)
print(t.get(1))         # 2
t.insert(4, 0)
print(t.getMin())       # 2
print(t.getMax())       # 0

t.insert(3, 7)
t.insert(2, 1)
print(t.getInorderKeys())  # [1, 2, 3, 4]
t.remove(1)
print(t.getInorderKeys())  # [2, 3, 4]

