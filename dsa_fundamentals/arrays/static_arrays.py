my_array = [1, 3, 5]


def insert_end(my_list, n, length, capacity):
    if length < capacity:
        my_list[length] = n


def insert_middle(my_list, i, n, length):
    for index in range(length - 1, i - 1, -1):
        my_list[index + 1] = my_list[index]
    my_list[i] = n


def remove_end(arr):
    if len(arr) > 0:
        arr[len(arr) - 1] = 0


def remove_middle(arr, i):
    for index in range(i + 1, len(arr)):
        arr[index - 1] = arr[index]


def print_array(arr):
    for i in range(len(arr)):
        print(arr[i])


print_array(my_array)
remove_end(my_array)
print_array(my_array)
