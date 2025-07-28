class TreeNode:
    def __init__(self, val):
        self.val = val
        self.left = None
        self.right = None

def can_reached_leaf(root):
    if not root or root.val == 0:
        return False
    if not root.left and not root.right:
        return True
    if can_reached_leaf(root.left):
        return True
    if can_reached_leaf(root.right):
        return True
    return False

def leaf_path(root, path):
    if not root or root.val == 0:
        return False
    path.append(root.val)

    if not root.left and not root.right:
        return True
    if leaf_path(root.left, path):
        return True
    if leaf_path(root.right, path):
        return True
    path.pop()
    return False