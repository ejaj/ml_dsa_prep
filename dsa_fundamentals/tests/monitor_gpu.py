import GPUtil
import time
def monitor_gpu(interval=2, duration=10):
    start_time = time.time()
    while time.time() - start_time < duration:
        gpus = GPUtil.getGPUs()
        for gpu in gpus:
            print(f"GPU {gpu.id} - Load: {gpu.load*100:.1f}%, Mem: {gpu.memoryUsed}/{gpu.memoryTotal}MB")
        time.sleep(interval)
monitor_gpu()

from multiprocessing import Process, Queue
import time

def gpu_benchmark(gpu_id, queue):
    time.sleep(2)
    queue.put((gpu_id, f"GPU, {gpu_id} done"))
q = Queue()
process = []
for i in range(2):
    p = Process(target=gpu_benchmark, args=(i, q))
    p.start()
    process.append(p)

for p in process:
    p.join()

while not q.empty():
    print(q.get())

import asyncio
import aiohttp

async def fetch_url(session, url):
    async with session.get(url) as response:
        text = await response.text()
        print(f"{url[:30]}...: {len(text)} chars")

async def main():
    urls = ["https://example.com"] * 5
    async with aiohttp.ClientSession() as session:
        tasks = [fetch_url(session, url) for url in urls]
        await asyncio.gather(*tasks)
asyncio.run(main())

import subprocess
def run_benchmark(cmd):
    print(f"Running: {cmd}")
    start = time.time()
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    duration = time.time() - start
    print(f"Finished in {duration:.2f} seconds")
    print("Output: ", result.stdout)
    return duration, result.stdout
run_benchmark("python my_model_infer.py")
