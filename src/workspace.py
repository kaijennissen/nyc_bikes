import argparse
import glob
import json
import logging
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import List, Tuple, Union

import coloredlogs
import googlemaps
import numpy as np
import pandas as pd
import requests
from jax import vmap

from data_prep import haversine

name_id_files = glob.glob("data/interim/" + "[0-9]" * 6 + "_name_id.feather")
dfs_name_id = [pd.read_feather(file_) for file_ in name_id_files]
df_names = pd.concat(dfs_name_id).drop_duplicates()
df_names.loc[df_names["station_id"].duplicated(keep=False), :].sort_values(
    "station_name"
)

df_replace = df_names.loc[
    df_names["station_name"].duplicated(keep=False), :
].sort_values("station_name")
df_replace["int"] = df_replace["station_id"].apply(isinstance, args=(int,)).astype(str)
df_replace.pivot(index="station_name", columns="int")
df_int = df_names.loc[
    df_names["station_id"].apply(isinstance, args=(int,)).astype(bool), :
]
df_notint = df_names.loc[
    ~df_names["station_id"].apply(isinstance, args=(int,)).astype(bool), :
]

df_mapping = df_notint.merge(
    df_int, how="left", on="station_name", suffixes=("_notint", "_int")
)

dict_mapping = (
    df_mapping.set_index("station_id_notint")
    .loc[:, ["station_id_int"]]
    .to_dict()["station_id_int"]
)
df_names["station_id_new"] = df_names["station_id"]
df_names["station_id_new"] = df_names["station_id_new"].replace(dict_mapping)


df_x = (
    pd.concat([df_ for df_ in dfs_name_id if df_.station_id.dtype == "int64"])
    .drop_duplicates()
    .reset_index(drop=True)
)

df_x.loc[df_x.station_id.duplicated(keep=False), :].sort_values("station_name")

id_loc_files = glob.glob("data/interim/" + "[0-9]" * 6 + "_id_loc.feather")
dfs_id_loc = [pd.read_feather(file_) for file_ in id_loc_files]

df = pd.concat(dfs_id_loc).drop_duplicates()
df = df.query("lat != 0 and lng != 0")

df.loc[df.station_id]

df_int = df.loc[df["station_id"].apply(isinstance, args=(int,)), :]
df_notint = df.loc[df["station_id"].apply(isinstance, args=(str,)), :]

lat, lng = 40.790179, -73.972889


def min_distance(df, lat, lng):
    df_int["distance"] = haversine(df_int["lat"], df_int["lng"], lat, lng)
    return df["distance"].idxmin()


df_unique_connections = df.loc[
    :, ["start_lat", "start_lng", "end_lat", "end_lng"]
].drop_duplicates()


def get_osrm_distance(lat1, lng1, lat2, lng2):
    r = requests.get(
        f"http://router.project-osrm.org/route/v1/car/{lng1},{lat1};{lng2},{lat2}?overview=false"
        ""
    )
    routes = json.loads(r.content)
    route_1 = routes.get("routes")[0]
    duration = route_1["duration"]
    distance = route_1["distance"]
    return distance, duration


API_key = open("credentials/google_api").readline().strip()
gmaps = googlemaps.Client(key=API_key)


def get_distance(lat1, lng1, lat2, lng2) -> Tuple[float, float]:
    dist = (
        gmaps.distance_matrix(
            origins=(lat1, lng1),
            destinations=(lat2, lng2),
            mode="bicycling",
            units="metric",
        )
        .get("rows")[0]
        .get("elements")[0]
    )

    if dist.get("status") == "OK":
        return dist.get("distance").get("value"), dist.get("duration").get("value")
    else:
        return np.nan, np.nan


df_subset = df.head(100)
df_subset[["google_distance", "google_duration"]] = df_subset.apply(
    lambda x: get_distance(x["start_lat"], x["start_lng"], x["end_lat"], x["end_lng"]),
    axis=1,
    result_type="expand",
)

df_subset[["osrm_distance", "osrm_duration"]] = df_subset.apply(
    lambda x: get_osrm_distance(
        x["start_lat"], x["start_lng"], x["end_lat"], x["end_lng"]
    ),
    axis=1,
    result_type="expand",
)
df_subset["distance"] = df_subset.osrm_distance - df_subset.google_distance
