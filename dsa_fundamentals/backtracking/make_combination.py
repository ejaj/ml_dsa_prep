def combine(n, k):
    result = []
    def make_combination(start, current):
        if len(current) == k:
            result.append(current[:])
            return
        for number in range(start, n+1):
            current.append(number)
            make_combination(number+1, current)
            current.pop()
    make_combination(1, [])
    return result
combine(4, 2)
