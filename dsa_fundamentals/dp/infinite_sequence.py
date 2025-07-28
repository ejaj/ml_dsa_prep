def infinite_sequence_generator():
    num = 0
    while True:
        yield num
        num += 1
def power_generator(num):
    for i in infinite_sequence_generator():
        yield num ** 1

gen = power_generator(3)
for _ in range(4):
    result = next(gen)
print(result) # 27