def longest_increasing_streak(data):
    if not data:
        return 0
    max_streak = 1
    current_streak = 1
    for i in range(1, len(data)):
        if data[i] > data[i-1]:
            current_streak += 1
            max_streak = max(max_streak, current_streak)
        else:
            current_streak = 1
    return max_streak

def get_longest_streak(data):
    if not data:
        return []
    
    best = []
    current = [data[0]]
    for i in range(1, len(data)):
        if data[i] > data[i-1]:
            current.append(data[i])
        else:
            if len(current) > len(best):
                best = current
            current = [data[i]]
    return best if len(best) > len(current) else current


def fibonacci_up_to(n):
    a, b  = 0, 1
    while a <= n:
        yield a
        a,b = b, a+b
for num in fibonacci_up_to(20):
    print(num)

def moving_average(k):
    window = []
    total = 0
    while True:
        num = yield total/len(window) if window else 0
        window.append(num)
        total += num
        if len(window) > k:
            total -= window.pop(0)




















