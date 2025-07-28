def greet(name):
    def format_name(n):
        return n.strip().title()
    return f"Hellom {format_name(name)}"
print(greet("  kazi ejajul  "))  # Hello, Kazi Ejajul

# With closure
def outer(msg):
    def inner():
        print(f"Message is : {msg}")
    return inner

my_func = outer("Hello from outer!")
my_func()  # Output: Message is: Hello from outer!

# function factory
def power_factory(n):
    def power(x):
        return x ** n
    return power
square = power_factory(2)
cube = power_factory(3)

print(square(2))
print(cube(2))