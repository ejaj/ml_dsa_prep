from multiprocessing import Process, Queue

def producer(q):
    for i in range(5):
        q.put(i)
def consumer(q):
    for _ in range(5):
        item = q.get()
        print("Got:", item)
if __name__ == "__main__":
    q = Queue()
    p1 = Process(target=producer, args=(q,))
    p2 = Process(target=consumer, args=(q,))
    p1.start()
    p2.start()
    p1.join()
    p2.join()

    # p.join(timeout=2)
    # if p.is_alive():
    #     print("Timeout: Terminating process")
    #     p.terminate()