from abc import ABC

class Base(ABC):
    def h(x):
        return 7
    @staticmethod
    def f():
        return 6

class Foo(Base):
    @staticmethod
    def f():
        return 5

    @classmethod
    def g(cls):
        return 8

    def h(self, x):
        return x.g()

    def __init__(self, x):
        self.x = x

res = Foo(6).h(Foo) + Foo.f()
print(res)