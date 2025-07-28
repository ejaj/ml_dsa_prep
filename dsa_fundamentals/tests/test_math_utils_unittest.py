import unittest
from math_utils import add

class TestMathUtils(unittest.TestCase):
    def test_add_positive(self):
        self.assertEqual(add(3,4), 7)
    def test_add_negative(self):
        self.assertEqual(add(-2, -5), -7)

    def test_add_zero(self):
        self.assertEqual(add(0, 5), 5)

if __name__ == '__main__':
    unittest.main()