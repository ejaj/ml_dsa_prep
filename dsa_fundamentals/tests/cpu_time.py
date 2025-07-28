import cProfile
def expensive_function():
    sum([i**2 for i in range(10000000)])
cProfile.run('expensive_function()')