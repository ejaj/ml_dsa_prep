import asyncio
import random

async def task(name):
    print(f"Start {name}")
    await asyncio.sleep(2)
    print(f"End {name}")


async def main():
    await asyncio.gather(task("A"), task("B"))

asyncio.run(main())


async def fetch_data(i):
    delay = random.randint(1, 3)
    await asyncio.sleep(delay)
    print(f"Fetched data {i} in {delay}s")

async def main():
    tasks = [fetch_data(i) for i in range(5)]
    asyncio.gather(*tasks)

asyncio.run(main)

async def say(text, delay):
    await asyncio.sleep(delay)
    print(f"{text} done after {delay}s")

async def main():
    tasks = [asyncio.create_task(say("A", 1)),
             asyncio.create_task(say("B", 3))]
    done, pending = await asyncio.wait(tasks, timeout=2)
    print("Done tasks:", done)
    print("Pending tasks:", pending)

asyncio.run(main())


async def long_task():
    await asyncio.sleep(0)
    return "Done"

async def main():
    try:
        result = await asyncio.wait_for(long_task(), timeout=2)
        print(result)
    except asyncio.TimeoutError:
        print("Timed out !")
asyncio.run(main())