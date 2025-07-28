import time
from functools import wraps

def log_and_time(tag="INFO"):
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            print(f"[{tag}] Calling function: {func.__name__}")
            print(f"[{tag}] Arguments: args={args}, kwargs={kwargs}")
            start = time.time()
            result = func(*args, **kwargs)
            end = time.time()
            print(f"[{tag}] Finished in {end - start:.4f} seconds")
            return result
        return wrapper
    return decorator

@log_and_time(tag="DEBUG")
def slow_add(a, b):
    time.sleep(1)  
    return a + b

@log_and_time(tag="TASK")
def greet(name, age=None):
    print(f"Hello {name}, age: {age}")

result = slow_add(3, 7)
greet("Alice", age=30)