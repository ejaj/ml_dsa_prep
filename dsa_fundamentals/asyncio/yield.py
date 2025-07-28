def count_up_to(max):
    num = 1
    while num <= max:
        yield num
        num += 1
for number in count_up_to(3):
    print(number)


gen = count_up_to(3)
print(next(gen))
print(next(gen))

def fib(limit):
    a,b = 0,1
    while a < limit:
        yield a
        a, b = b ,a+b

for n in fib(10):
    print(n)