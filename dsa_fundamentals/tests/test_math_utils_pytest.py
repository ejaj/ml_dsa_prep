from math_utils import add
def test_add_positive():
    assert add(3,4) == 7
def test_add_negative():
    assert add(-2,-5) == -7
def test_add_zero():
    assert add(0, 5) == 5
