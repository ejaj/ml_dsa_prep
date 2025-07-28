import threading
def worker():
    print("Thread is working...")

def print_msg(msg):
    print(f"Message: {msg}")

t = threading.Thread(target=worker)
t.start()
t.join()

threads = []
for i in range(5):
    t = threading.Thread(target=print_msg, args=(f"Hello {i}",))
    t.start()
    threading.append(t)

for t in threads:
    t.join()

from queue import Queue
def worker(q):
    while not q.empty():
        item = q.get()
        print(f"Processing {item}")
        q.task_done(())
q = Queue()
for i in range(10):
    q.put(i)
for _ in range(3):
    t = threading.Thread(target=worker, args=(q,))
    t.start()
q.join()