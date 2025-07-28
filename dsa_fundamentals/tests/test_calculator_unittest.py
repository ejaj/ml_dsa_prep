from calculator import add
import unittest

class TestCalculator(unittest.TestCase):
    def test_add(self):
        self.assertEqual(add(2, 3), 5)
        self.assertEqual(add(-1, 1), 0)
        self.assertNotEqual(add(1,1), 3)
if __name__ == '__main__':
    unittest.main()