def is_power_of_two(n):
    if n <= 0:
        return False
    return (n & (n-1)) == 0

test_numbers = [1, 2, 3, 4, 5, 8, 10, 16, 18, 32]

for num in test_numbers:
    if is_power_of_two(num):
        print(f"{num} is a power of two.")
    else:
        print(f"{num} is NOT a power of two.")
