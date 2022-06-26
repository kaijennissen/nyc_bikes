import argparse
import glob
import logging
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime
from pathlib import Path
from typing import List, Union

import coloredlogs
import dask.dataframe as dd
import googlemaps
import holidays
import numpy as np
import pandas as pd
import requests
from geopy.extra.rate_limiter import RateLimiter
from geopy.geocoders import Nominatim

from helpers import read_from_dir, save_to_dir

logger = logging.getLogger("data_prep")
coloredlogs.install(
    fmt="%(asctime)s - %(name)s - %(process)s - %(levelname)s - %(message)s",
    level="DEBUG",
)

expected_cols = [
    "trip_duration",
    "start_time",
    "stopt_time",
    "start_station_id",
    "start_station_name",
    "start_lat",
    "start_lng",
    "end_station_id",
    "end_station_name",
    "end_lat",
    "end_lng",
    "bike_id",
    "usertype",
    "birth_year",
]


def haversine(lat1, lon1, lat2, lon2):
    lon1 = np.radians(lon1)
    lon2 = np.radians(lon2)
    lat1 = np.radians(lat1)
    lat2 = np.radians(lat2)

    h = (
        np.sin((lat2 - lat1) / 2.0) ** 2
        + np.cos(lat1) * np.cos(lat2) * np.sin((lon2 - lon1) / 2.0) ** 2
    )
    miles = 3959 * (2 * np.arcsin(np.sqrt(h)))
    return miles * 1.60934 * 1000


def process_all(files: List) -> pd.DataFrame:
    futures_list = []
    results = []
    with ProcessPoolExecutor(max_workers=8) as executor:
        # return executor.map(data_prep, files, timeout=600)
        for file_ in files:
            futures = executor.submit(data_prep, file_)
            futures_list.append(futures)

        for future in futures_list:
            try:
                result = future.result(timeout=600)
                results.append(result)
            except Exception:  # pylint: disable=broad-except
                results.append(None)  # type:ignore

    return results


def request_openweather(lat, lng, time):
    API_KEY = open("credentials/openweather_api").read().strip()

    url = f"https://api.openweathermap.org/data/3.0/onecall/timemachine?lat={lat}&lon={lng}&dt={int(time)}&appid={API_KEY}"

    response = requests.get(url)
    if response.status_code == 200:
        resu = response.json()["data"][0]
        weather_dict = resu.pop("weather")[0]
        resu.update(weather_dict)
        return resu


def make_weather_df(start_date, end_date, lat, lng) -> pd.DataFrame:
    df = pd.DataFrame(
        {
            "datetime": pd.date_range(
                start=start_date,
                end=end_date,
                freq="H",
                normalize=True,
                inclusive="left",
            )
        }
    )
    df["lat"] = lat
    df["lng"] = lng
    df["timestamp"] = (
        (df.datetime - datetime.strptime("1970-01-01 00:00:00", "%Y-%m-%d %H:%M:%S"))
        .dt.total_seconds()
        .astype(np.int64)
    )

    weather_list = []
    for _, row in df.iterrows():
        weather_list.append(
            request_openweather(lat=row["lat"], lng=row["lng"], time=row["timestamp"])
        )
    temp_cols = [
        "temp",
        "feels_like",
    ]
    df_weather = pd.DataFrame(weather_list)
    for col in temp_cols:
        df_weather[col] = df_weather[col] - 273.15
    return df_weather


def make_google_distance(df):
    API_key = open("credentials/google_api").read().strip()
    gmaps = googlemaps.Client(key=API_key)

    def request_distance(start_lat, start_lng, end_lat, end_lng):
        result = gmaps.distance_matrix(
            (start_lat, start_lng),
            (end_lat, end_lng),
            mode="bicycling",
        )
        if result.get("status") == "OK":
            result_distance = result["rows"][0]["elements"][0]["distance"]["value"]
            result_time = result["rows"][0]["elements"][0]["duration"]["value"]
        else:
            result_distance = np.nan
            result_time = np.nan
        return {"distance": result_distance, "time": result_time}

    df_trips = df.loc[
        :, ["start_lat", "start_lng", "end_lat", "end_lng"]
    ].drop_duplicates()

    trips = []
    for _, row in df.head(2000).iterrows():
        trips.append(
            request_distance(
                row["start_lat"],
                row["start_lng"],
                row["end_lat"],
                row["end_lng"],
            )
        )

    df_distance = pd.DataFrame(trips)
    df_distance = df_trips.join(df_distance)
    return df_distance


def make_station_df(df):
    logger.info("Preparing Station df")
    df = df.filter(
        [
            "start_station_id",
            "end_station_id",
            "start_station_name",
            "end_station_name",
            "start_lat",
            "start_lng",
            "end_lat",
            "end_lng",
        ]
    ).drop_duplicates()
    df_station_start = (
        df.filter(["start_station_id", "start_station_name", "start_lat", "start_lng"])
        .rename(
            {
                "start_station_id": "station_id",
                "start_station_name": "station_name",
                "start_lat": "lat",
                "start_lng": "lng",
            },
            axis=1,
        )
        .drop_duplicates()
    )
    df_station_end = (
        df.filter(["end_station_id", "end_station_name", "end_lat", "end_lng"])
        .rename(
            {
                "end_station_id": "station_id",
                "end_station_name": "station_name",
                "end_lat": "lat",
                "end_lng": "lng",
            },
            axis=1,
        )
        .drop_duplicates()
    )
    df_station = pd.concat([df_station_start, df_station_end]).drop_duplicates()
    logger.info("Prepared station DataFrame.")
    return df_station.dropna()


def data_prep(
    file_: Union[str, Path],
) -> str:

    if isinstance(file_, str):
        file_ = Path(file_)

    logger.info(f"Processing file: {file_}")
    yw = Path(file_).stem[:-18]

    df = pd.read_csv(file_, nrows=5)
    if "starttime" in df.columns:
        df = pd.read_csv(file_, parse_dates=["starttime", "stoptime"], low_memory=False)
    elif "started_at" in df.columns:
        df = pd.read_csv(
            file_, parse_dates=["started_at", "ended_at"], low_memory=False
        )

    elif "Start Time" in df.columns:
        df = pd.read_csv(
            file_, parse_dates=["Start Time", "Stop Time"], low_memory=False
        )
    else:
        logger.warning(f"Unknown columns: {df.columns.tolist()}")

    df.columns = df.columns.str.lower()
    col_mapping = {
        # time
        "started_at": "start_time",
        "ended_at": "end_time",
        "start time": "start_time",
        "stop time": "end_time",
        # station id
        "start station id": "start_station_id",
        "end station id": "end_station_id",
        # station name
        "start station name": "start_station_name",
        "end station name": "end_station_name",
        # station latitude
        "start station latitude": "start_lat",
        "end station latitude": "end_lat",
        # station longitude
        "start station longitude": "start_lng",
        "end station longitude": "end_lng",
        # trip duration
        "trip duration": "trip_duration",
        "tripduration": "trip_duration",
        # else
        "member_casual": "usertype",
        "birth year": "birth_year",
        "bike id": "bike_id",
        "bikeid": "bike_id",
        "user type": "usertype",
    }

    df = df.rename(
        col_mapping,
        axis=1,
    )
    # remove columns with missing values
    df = df.query("start_lat != 0 and end_lng != 0 and start_lng != 0 and end_lng != 0")
    # check for expected columns
    for col in expected_cols:
        if col not in df.columns:
            df[col] = np.nan

    if not set(col_mapping.values()).issubset(set(df.columns.tolist())):
        logger.warning(
            f"Missing columns: {set(df.columns.tolist())-set(col_mapping.values())}"
        )

    df["trip_duration"] = (df["stoptime"] - df["starttime"]).dt.seconds
    df["weekday"] = df["starttime"].dt.isocalendar().day
    df["month"] = df["starttime"].dt.month
    df["year"] = df["starttime"].dt.year
    df["hour"] = df["starttime"].dt.hour
    # df["date"] = df["starttime"].dt.strftime("%Y-%m-%d")
    df["date"] = df.starttime.dt.date
    df["date"] = pd.to_datetime(df["date"], format="%Y-%m-%d")

    df["trip_duration_min"] = df["trip_duration"] / 60
    df["trip_duration_hour"] = df["trip_duration"] / 3600

    df["roundtrip"] = (df["start_station_id"] == df["end_station_id"]).astype(np.int16)
    df["simple_distance"] = haversine(
        df["start_lat"].values,
        df["start_lng"].values,
        df["end_lat"].values,
        df["end_lng"].values,
    )
    df = add_holidays(df)

    df.to_parquet(f"data/interim/{yw}.parquet")
    logger.info(f"Processed file: {file_}")

    return f"Processed file: {file_}"


def make_station_geolocation(df: pd.DataFrame) -> pd.DataFrame:
    geolocator = Nominatim(user_agent="bikeshare")
    reverse = RateLimiter(geolocator.reverse, min_delay_seconds=1, max_retries=0)

    df = df.dropna().astype({"station_id": np.int64}).reset_index(drop=True)

    locations = []
    for _, row in df.iterrows():
        locations.append(reverse(f"{row['lat']}, {row['lng']}").raw["address"])

    df_loc = pd.DataFrame(locations)
    df = df.join(df_loc, how="left")
    return df


def add_holidays(df: pd.DataFrame) -> pd.DataFrame:
    non_working_days = [
        "New Year's Day",
        "Memorial Day",
        "Independence Day",
        "Labor Day",
        "Thanksgiving Day",
        "Christmas Eve",
        "Christmas Day",
    ]

    holidays_ = holidays.country_holidays(country="US", years=2018, state="NY")
    holidays_.update(
        {
            datetime.strptime("2018-11-23", "%Y-%m-%d"): "Thanksgiving",
            datetime.strptime("2018-12-24", "%Y-%m-%d"): "Christmas Eve",
            datetime.strptime("2018-12-31", "%Y-%m-%d"): "Silvester",
        }
    )
    df_holidays = pd.DataFrame(holidays_.items(), columns=["date", "holiday_name"])

    df = df.merge(df_holidays, how="left", on="date")
    df["holiday"] = (df.holiday_name.fillna(np.nan)).notna().astype(int)
    df["working_day"] = 1

    df.loc[
        (df.holiday_name.isin(non_working_days)) | (df.weekday >= 6),
        "working_day",
    ] = 0

    return df


def main(
    make_data_prep: bool = False,
    multi_process: bool = False,
    geolocation: bool = False,
    weather: bool = False,
    distance: bool = False,
):
    dir_in = Path("data/raw")
    dir_out = Path("data/interim")

    if make_data_prep:
        files = sorted(glob.glob(str(dir_in) + "/*2018**-citibike-tripdata.csv"))

        if multi_process:
            resu = process_all(files)
        else:
            resu = [data_prep(file_) for file_ in files]

    if geolocation:
        # df = dd.read_parquet(dir_out / "*2018**.parquet").compute()
        # df = df.reset_index(drop=True)
        # df_station = make_station_df(df)
        # df_station = make_station_geolocation(df_station)
        # df_station.to_parquet("data/processed/df_station.parquet")

        # df = dd.read_parquet(dir_out / "2018**.parquet").compute()
        files = sorted(glob.glob(str(dir_out) + "/*2018**.parquet"))
        df = pd.concat([pd.read_parquet(file_) for file_ in files]).reset_index(
            drop=True
        )

        df_station = pd.read_parquet("data/processed/df_station.parquet")
        for suffix in ["start", "end"]:
            df_merge = df_station.loc[
                :,
                ["station_id", "station_name", "lat", "lng", "city", "suburb"],
            ]
            df_merge = df_merge.rename(
                {col: suffix + "_" + col for col in df_merge.columns},
                axis=1,
            )
            df = df.merge(df_merge, how="left")

        df = df.query(
            "start_city == 'City of New York' and end_city == 'City of New York'"
        )
        df.loc[df.start_suburb.isna(), "start_suburb"] = "Manhatten"
        df.loc[df.end_suburb.isna(), "end_suburb"] = "Manhatten"
        save_to_dir(df, outdir="data/processed")

        # Debugging
        # df12 = df.query("month==12")
        # grouped = df12.groupby(["date"])
        # for day, df_m in grouped:
        #     outfile = f"data/processed/test/{day}.parquet"
        #     logger.info(f"writing to {outfile}")
        #     df_m.to_parquet(outfile)
        # files = sorted(glob.glob("data/processed/test/2018**.parquet"))
        # ls_ = []
        # for file_ in files:
        #     logger.info(f"Reading: {file_}")
        #     df = pd.read_parquet(file_)
        #     ls_.append(df)
        # logger.info("Finished reading!")

        # df.to_parquet("data/processed/df_nyc.parquet")

    if weather:
        df = read_from_dir("data/processed")
        # ls_ = []
        # files = sorted(glob.glob("data/processed/2018**.parquet"))
        # for file_ in files:
        #     logger.info(f"Reading: {file_}")
        #     df = pd.read_parquet(file_)
        #     ls_.append(df)
        # logger.info("Finished reading!")
        # df = pd.concat(ls_)
        # logger.info("Combined all files!")

        # df = pd.concat([pd.read_parquet(file_) for file_ in files]).reset_index(drop=True)

        # df = pd.read_parquet("data/processed/df_nyc.parquet")
        df["datetime"] = df["starttime"].dt.strftime("%Y-%m-%d %H:00")

        df_weather = pd.read_parquet(
            "data/external/weather_history_2018.parquet"
        ).reset_index()
        df_weather["datetime"] = df_weather["date"].dt.strftime("%Y-%m-%d %H:00")
        df_weather = df_weather.drop("date", axis=1)
        df_weather = df_weather.groupby("datetime").agg(
            temp=("temp", np.mean),
            visibility=("visibility", np.mean),
            dew_point=("dew_point", np.mean),
            feels_like=("feels_like", np.mean),
            temp_min=("temp_min", np.mean),
            temp_max=("temp_max", np.mean),
            pressure=("pressure", np.mean),
            sea_level=("sea_level", np.mean),
            grnd_level=("grnd_level", np.mean),
            humidity=("humidity", np.mean),
            wind_speed=("wind_speed", np.mean),
            rain_1h=("rain_1h", np.mean),
            rain_3h=("rain_3h", np.mean),
            snow_1h=("snow_1h", np.mean),
            snow_3h=("snow_3h", np.mean),
            weather_main=("weather_main", "first"),
        )
        df_weather["weather_main"] = df_weather["weather_main"].replace(
            {"Haze": "Mist", "Fog": "Mist", "Drizzle": "Rain"}
        )
        df = df.merge(df_weather, how="left", on="datetime")

        # df.to_parquet("data/processed/df_nyc.parquet")
        save_to_dir(df, outdir="data/output")
        # grouped = df.groupby(["year", "month"])
        # for (year, month), df_m in grouped:
        #     outfile = f"data/processed/{year}{str(month).zfill(2)}.parquet"
        #     logger.info(f"writing to {outfile}")
        #     df_m.to_parquet(outfile)

    # if False:
    #     df = pd.read_parquet("data/processed/df_nyc.parquet")
    #     df_unique_trips = df.loc[
    #         :, ["start_lat", "start_lng", "end_lat", "end_lng"]
    #     ].drop_duplicates()
    #     df_distance = make_openmap_distance(df_unique_trips)
    #     df_distance = make_google_distance(df_unique_trips)
    #     df_distance.to_parquet("data/processed/df_distance.parquet")


if __name__ == "__main__":
    argparser = argparse.ArgumentParser()
    argparser.add_argument("--data-prep", action="store_true", help="Date Preparation")
    argparser.add_argument("--mp", action="store_true", help="Multiprocessing")
    argparser.add_argument("--geolocation", action="store_true", help="Geolocation")
    argparser.add_argument("--weather", action="store_true", help="Weather")
    argparser.add_argument("--distance", action="store_true", help="distance")
    args = argparser.parse_args()

    main(
        make_data_prep=args.data_prep,
        multi_process=args.mp,
        geolocation=args.geolocation,
        weather=args.weather,
        distance=args.distance,
    )
