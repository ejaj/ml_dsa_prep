import asyncio
import aiohttp
import time

async def fetch(url):
    async with aiohttp.ClientSession() as session:
        async with session.get(url) as resp:
            return await resp.text()

async def fake_api_call(name: str, delay: int):
    print(f"{name} started")
    await asyncio.sleep(delay)
    print(f"{name} finished")
    return f"{name} response"

async def main():
    # urls = ['https://example.com', 'https://httpbin.org/get']
    # start = time.perf_counter()
    # results = await asyncio.gather(*(fetch(u) for u in urls))
    # end = time.perf_counter()
    # for r in results:
    #     print(r[:100])
    # print("\n--- Response Preview ---")
    # for i, r in enumerate(results, 1):
    #     print(f"Response {i}: {r[:100]}...\n")

    # print(f"Total time taken: {end - start:.2f} seconds")
    start = time.perf_counter()
    tasks = [
        fake_api_call("API 1", 2),
        fake_api_call("API 2", 1),
        fake_api_call("API 3", 3),
        fake_api_call("API 4", 2),
        fake_api_call("API 5", 1),
    ]
    results = await asyncio.gather(*tasks)
    end = time.perf_counter()
    print("\nAll APIs completed.")
    print("Results:", results)
    print(f"Total time taken: {end - start:.2f} seconds")


asyncio.run(main())
