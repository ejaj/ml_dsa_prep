def my_decorator(func):
    def wrapper():
        print("Before function call")
        func()
        print("After function call")
    return wrapper

@my_decorator
def say_hello():
    print("hello")
say_hello()

# say_hello = my_decorator(say_hello), witout @my_decorator

def debug(func):
    def wrapper(*args, **kwargs):
        print(f"Calling {func.__name__} with {args} {kwargs}")
        result = func(*args, **kwargs)
        print(f"{func.__name__} returned {result}")
        return result
    return wrapper
@debug
def add(a, b):
    return a+b
add(3,4)



def debug_con(log=True):
    def decorator(func):
        def wrapper(*args, **kwargs):
            if log:
                print(f"[DEBUG] Calling {func.__name__} with {args} {kwargs}")
            result = func(*args, **kwargs)
            result = func(*args, **kwargs)
            if log:
                print(f"[DEBUG] {func.__name__} returned {result}")
            return result
        return wrapper
    return decorator

@debug(log=True)
def add(a, b):
    return a + b

@debug(log=False)
def multiply(a, b):
    return a * b

print(add(3, 4))       # Logs everything
print(multiply(3, 4))  # Silent










