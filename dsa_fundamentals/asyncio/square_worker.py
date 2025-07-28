from multiprocessing import Pool
def square(n):
    return n*n


def count_lines(file_path):
    with open(file_path) as f:
        return file_path, sum(1 for _ in f)


if __name__ == "__main__":
    numbers = [1,2,3,4,5]
    with Pool(processes=3) as pool:
        results = pool.map(square, numbers)
    print("Squres:", results)
    files = ["file1.txt", "file2.txt", "file3.txt"]
    with Pool() as pool:
        results = pool.map(count_lines, files)
    print(results)