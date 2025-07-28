from multiprocessing import Process, Value

def increment(counter):
    for _ in range(1000):
        with counter.get_lock():
            counter.value += 1

if __name__ == "__main__":
    counter = Value('i', 0)
    processes = [Process(target=increment, args=(counter,)) for _ in range(5)]

    for p in processes:
        p.start()
    for p in processes:
        p.join()
    print("Final counter value:", counter.value)  # Expected: 5000

