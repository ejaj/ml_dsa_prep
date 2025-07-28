import time, psutil, logging, json
from pathlib import Path

def run_benchmark():
    start = time.time()
    process = psutil.Process()
    result = sum(i**2 for i in range(10**6))
    end = time.time()
    cpu_time = round(end - start, 2)
    mem_usage = round(process.memory_info().rss / 1024 / 1024, 2)
    
    return {"cpu_time": cpu_time, "memory_usage": mem_usage, "result": result}

def save_results(metrics):
    Path("results").mkdir(exist_ok=True)
    with open("results/benchmark_results.json", "w") as f:
        json.dump(metrics, f, indent=2)

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    logging.info("Starting benchmark...")
    metrics = run_benchmark()
    logging.info(f"Results: {metrics}")
    save_results(metrics)


