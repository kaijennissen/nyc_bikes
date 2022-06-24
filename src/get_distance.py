import argparse
import asyncio
import json
import logging
import time
from concurrent.futures import ThreadPoolExecutor
from multiprocessing import Pool

import coloredlogs
import numpy as np
import pandas as pd
import requests
from aiohttp import ClientSession

logger = logging.getLogger("data_prep")
coloredlogs.install(
    fmt="%(asctime)s - %(name)s - %(process)s - %(levelname)s - %(message)s",
    level="INFO",
)


def get_osrm_distance(idx, lat1, lng1, lat2, lng2):
    r = requests.get(
        f"http://router.project-osrm.org/route/v1/car/{lng1},{lat1};{lng2},{lat2}?overview=false"
        ""
    )
    routes = json.loads(r.content)
    if routes.get("code") == "Ok":
        route_1 = routes.get("routes")[0]
        duration = route_1["duration"]
        distance = route_1["distance"]
        return idx, distance, duration
    else:
        return idx, np.nan, np.nan


async def fetch(url, idx, session):
    async with session.get(url) as response:
        return await response.read()


def extract_results(response):
    try:
        routes = json.loads(response)
    except:
        breakpoint()
    if routes.get("code") == "Ok":
        route_1 = routes.get("routes")[0]
        duration = route_1["duration"]
        distance = route_1["distance"]
        return distance, duration
    else:
        return np.nan, np.nan


async def run(trips):
    # url = "http://localhost:8080/{}"
    # url=f"http://router.project-osrm.org/route/v1/car/{lng1},{lat1};{lng2},{lat2}?overview=false"
    # url = (
    #     "http://router.project-osrm.org/route/v1/car/{},{};{},{}?overview=false"
    # )

    tasks = []
    # Fetch all responses within one Client session,
    # keep connection alive for all requests.
    async with ClientSession() as session:
        # for idx, *args in trips:
        for idx, lat1, lng1, lat2, lng2 in trips:
            url = f"http://router.project-osrm.org/route/v1/car/{lng1},{lat1};{lng2},{lat2}?overview=false"
            # task = asyncio.ensure_future(fetch(url.format(*args), idx, session))
            task = asyncio.ensure_future(fetch(url, idx, session))
            tasks.append(task)

        responses = await asyncio.gather(*tasks)
        # you now have all response bodies in this variable
        return responses


def main(mode: str = "sequential", n: int = 10):
    logger.info("Starting main!")
    df_trips = pd.read_parquet("data/processed/df_trips.parquet")
    # df = pd.read_parquet("data/processed/df_nyc.parquet")
    # df_trips = (
    #     df.loc[:, ["start_lat", "start_lng", "end_lat", "end_lng"]]
    #     .drop_duplicates()
    #     .reset_index(drop=True)
    # )
    # del df
    # df_trips.to_parquet("data/processed/df_trips.parquet")

    if n <= 0:
        n = df_trips.shape[0]

    ls_ = list(
        df_trips.head(n)[["start_lat", "start_lng", "end_lat", "end_lng"]].to_records()
    )
    ls_gen = (y for y in ls_)
    if mode == "asyncio":
        loop = asyncio.get_event_loop()
        future = asyncio.ensure_future(run(ls_gen))
        loop.run_until_complete(future)
        resu = future.result()
        osrm_resu = [extract_results(r) for r in resu]
        osrm_resu = [
            (a, b, c)
            for a, (b, c) in zip([i for i in range(len(osrm_resu))], osrm_resu)
        ]

    elif mode == "multiprocess":
        logger.info("Started multiprocessing!")
        start = time.time()
        ls_gen = (y for y in ls_)
        try:
            pool = Pool(10)  # Create a multiprocessing Pool
            osrm_resu = pool.starmap(get_osrm_distance, ls_gen)
        finally:  # To make sure processes are closed in the end, even if errors happen
            pool.close()
            pool.join()
        stop = time.time()
        logger.info(f"Time: {stop-start} seconds.")
        logger.info("Finished multiprocessing!")

    elif mode == "multithread":
        logger.info("Started multithreading!")
        futures_list = []
        osrm_resu = []
        with ThreadPoolExecutor(max_workers=10) as executor:
            for args in ls_gen:
                futures = executor.submit(lambda z: get_osrm_distance(*z), args)
                futures_list.append(futures)

        for future in futures_list:
            try:
                result = future.result(timeout=60)
                osrm_resu.append(result)
            except Exception:
                osrm_resu.append(None)

        logger.info("Finished multithreading!")

    elif mode == "sequential":
        logger.info("Started sequential!")

        start = time.time()
        osrm_resu = [get_osrm_distance(*entry) for entry in ls_gen]
        stop = time.time()
        print(f"Time: {stop-start}")
        logger.info("Finished sequential!")

    df_distance = pd.DataFrame(
        osrm_resu, columns=["index", "osrm_distance", "osrm_duration"]
    ).set_index("index")
    print(df_distance.head())
    df_distance.to_parquet("data/processed/df_distance.parquet")
    logger.info("Saves to data/processed/df_distance.parquet")
    logger.info("Finished main!")


if __name__ == "__main__":
    argparser = argparse.ArgumentParser()
    argparser.add_argument(
        "--mode", default="sequential", type=str, help="Multiprocessing"
    )
    argparser.add_argument("--n", default=10, type=int, help="size")
    args = argparser.parse_args()

    main(mode=args.mode, n=args.n)
