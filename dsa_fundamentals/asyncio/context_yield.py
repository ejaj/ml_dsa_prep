import time
from contextlib import contextmanager

def slow_function(seconds=2):
    time.sleep(seconds)
    return seconds

@contextmanager
def profiler_context():
    st_time = time.time()
    try:
        yield
    finally:
        end_time = time.time()
        print(f"Execution time: {end_time - st_time:.4f} seconds")
with profiler_context():
    reseult = slow_function()