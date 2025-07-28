arr = [1, 2, 3]
# Calculate:
# 1^2 % MOD = 1
# 2^3 % MOD = 8
# max is 8 at index 1 → return 2 (1-based)


def raisingPower(arr):
    MOD = 10**9 + 7
    max_val = -1
    min_index = -1
    for i in range(len(arr) - 1):  # stop at second last element
        base = arr[i]
        exponent = arr[i + 1]
        val = pow(base, exponent, MOD)

        if val > max_val:
            max_val = val
            min_index = i + 1  # convert 0-based to 1-based
        elif val == max_val:
            min_index = (min_index, i+1)
    return min_index

print(raisingPower(arr))
