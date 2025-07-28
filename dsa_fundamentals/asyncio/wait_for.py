import asyncio
import aiohttp
import time

async def fetch(session, url, timeout=3):
    try:
        async with session.get(url) as resp:
            print(f"Fetched {url} - Status: {resp.status}")
            return await resp.text()
    except asyncio.TimeoutError:
        print(f"Timeout while fetching: {url}")
        return None

async def fetch_with_timeout(session, url, timeout=3):
    try:
        return await asyncio.wait_for(fetch(session, url, timeout), timeout)
    except asyncio.TimeoutError:
        print(f"wait_for timeout reached for: {url}")
        return None

async def main():
    urls = [
        'https://example.com',
        'https://httpbin.org/get',
        'https://httpbin.org/delay/5',  # Will timeout
        'https://httpbin.org/uuid',
        'https://httpbin.org/headers'
    ]
    start = time.perf_counter()
    async with aiohttp.ClientSession() as session:
        tasks = [fetch_with_timeout(session, url, timeout=3) for url in urls]
        results = await asyncio.gather(*tasks)
    end = time.perf_counter()

    print("\n--- Response Preview ---")
    for i, r in enumerate(results, 1):
        print(f"Response {i}: {r[:100] if r else 'No response (timed out)'}\n")

    print(f"Total time taken: {end - start:.2f} seconds")

from asyncio import run, sleep, wait_for
from random import random
async def it_might_take_long():
    print(f"Doing something ...")
    time_to_sleep = random() * 2
    await sleep(time_to_sleep)
    print("Done !")
async def count_timeouts(coro, tries, timeout=1):
    timeout_count = 0
    for _ in range(tries):
        try:
            await wait_for(coro(), timeout=timeout)
        except Exception:
            timeout_count += 1
    print(f"Timeouts: {timeout_count}")


if __name__ == "__main__":
    asyncio.run(main())









