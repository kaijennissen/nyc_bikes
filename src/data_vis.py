import glob
import logging
import pickle
import time
from datetime import datetime
from pathlib import Path
from urllib import response

import coloredlogs
import dask.dataframe as dd
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.figure_factory as ff
import plotly.graph_objects as go
import requests
from matplotlib import markers

logger = logging.getLogger("data_vis")
coloredlogs.install(
    fmt="%(asctime)s - %(name)s - %(process)s - %(levelname)s - %(message)s",
    level="DEBUG",
)

px.set_mapbox_access_token(open("credentials/mapbox_token").read())
mapbox_access_token = open("credentials/mapbox_token").read().strip()

category_orders_month = [
    "Jan",
    "Feb",
    "Mar",
    "Apr",
    "May",
    "Jun",
    "Jul",
    "Aug",
    "Sep",
    "Oct",
    "Nov",
    "Dec",
]
category_orders_usertype = ["Subscriber", "Customer"]
category_orders_weekdays = [
    "Monday",
    "Tuesday",
    "Wednesday",
    "Thursday",
    "Friday",
    "Saturday",
    "Sunday",
]
COLORS24 = ["hsl(" + str(h) + ",50%" + ",50%)" for h in np.linspace(0, 360, 24)]
COLORS12 = ["hsl(" + str(h) + ",50%" + ",50%)" for h in np.linspace(0, 360, 12)]
COLORS4 = [COLORS12[0], COLORS12[4], COLORS12[6], COLORS12[9]]
COLORS2 = [COLORS12[4], COLORS12[9]]


def to_pickle(fig, filename: str) -> None:
    with open(f"images/{filename}.pickle", "wb") as f:
        pickle.dump(fig, f)


def from_pickle(filename: str):
    with open(f"images/{filename}.pickle", "rb") as f:
        fig = pickle.load(f)
    return fig


def make_plot1(df_plot1):
    logger.info("Started make_plot1...")
    # Plot 1: Trips per Hour by Day
    df_plot1 = (
        df.loc[:, ["starttime", "usertype", "hour"]]
        .set_index("starttime")
        .groupby(["usertype"])[["hour"]]
        .resample("H")
        .count()
        .rename(columns={"hour": "count"})
        .reset_index()
    )

    df_plot1["weekday"] = df_plot1["starttime"].dt.strftime("%A")
    df_plot1["hour"] = df_plot1["starttime"].dt.strftime("%H:%M")
    df_plot1 = df_plot1.reset_index(drop=True)

    fig = px.box(
        df_plot1,
        x="hour",
        y="count",
        color="hour",
        facet_row_spacing=0.01,
        facet_col_spacing=0.025,
        facet_row="weekday",
        facet_col="usertype",
        color_discrete_sequence=COLORS24,
        category_orders={
            "weekday": category_orders_weekdays,
            "usertype": category_orders_usertype,
        },
        points=False,
    )
    fig.update_yaxes(matches=None, showticklabels=True)
    fig.update_layout(
        hovermode="x unified", title_text="Trips per Hour by Day"
    )  # ,showlegend=False)

    fig.layout.yaxis1.matches = "y1"
    fig.layout.yaxis3.matches = "y1"
    fig.layout.yaxis5.matches = "y1"
    fig.layout.yaxis7.matches = "y1"
    fig.layout.yaxis9.matches = "y1"
    fig.layout.yaxis11.matches = "y1"
    fig.layout.yaxis13.matches = "y1"

    fig.layout.yaxis2.matches = "y2"
    fig.layout.yaxis4.matches = "y2"
    fig.layout.yaxis6.matches = "y2"
    fig.layout.yaxis8.matches = "y2"
    fig.layout.yaxis10.matches = "y2"
    fig.layout.yaxis12.matches = "y2"
    fig.layout.yaxis14.matches = "y2"
    to_pickle(fig, "plot1")
    logger.info("Finished make_plot1")


# Plot 2: Trips per Day over Year
def make_plot2(df):
    logger.info("Started make_plot2...")
    df_plot2 = (
        df.set_index("starttime")
        .groupby("usertype")[["hour"]]
        .resample("D")
        .count()
        .rename(columns={"hour": "count"})
        .reset_index()
    )
    df_plot2["month"] = df_plot2.starttime.dt.strftime("%b")  # month
    df_plot2["day"] = df_plot2.starttime.dt.dayofyear
    df_plot2["year"] = df_plot2.starttime.dt.year
    df_plot2 = df_plot2.reset_index(drop=True)

    fig = px.box(
        df_plot2,
        x="month",
        y="count",
        color="month",
        facet_row="year",
        facet_col="usertype",
        facet_row_spacing=0.001,
        color_discrete_sequence=COLORS12,
        category_orders={
            "month": category_orders_month,
            "usertype": category_orders_usertype,
        },
        points=False,
    )
    fig.update_yaxes(matches=None, showticklabels=True)
    fig.update_layout(
        boxgap=0.125,
        boxgroupgap=0.125,
    )
    to_pickle(fig, "plot2")
    logger.info("Finished make_plot2")


# Plot 3: Abs and Rel Trips per Day over Year
def make_plot3(df_plot3):
    logger.info("Started make_plot3...")
    df_plot3 = (
        df.set_index("starttime")
        .groupby("usertype")[["hour"]]
        .resample("D")
        .count()
        .rename(columns={"hour": "count"})
        .reset_index()
    )
    df_plot3["month"] = df_plot3.starttime.dt.strftime("%b")  # month
    df_plot3["day"] = df_plot3.starttime.dt.dayofyear
    df_plot3["year"] = df_plot3.starttime.dt.year
    df_plot3 = df_plot3.reset_index(drop=True)

    df_plot3 = df_plot3.merge(
        df_plot3.groupby("starttime").agg(count_total=("count", np.sum)),
        on="starttime",
        how="left",
    )
    df_plot3["rel_count"] = df_plot3["count"] / df_plot3["count_total"]
    df_plot3 = (
        df_plot3.drop(["count_total", "month", "day", "year"], axis=1)
        .rename(columns={"count": "abs", "rel_count": "rel"})
        .melt(
            id_vars=["usertype", "starttime"],
            value_vars=["abs", "rel"],
            value_name="count",
            var_name="count_type",
        )
    )

    fig = px.line(
        df_plot3,
        x="starttime",
        y="count",
        color="usertype",
        hover_data={
            "count": True,
            "usertype": False,
            "starttime": False,
            "count_type": False,
        },
        color_discrete_sequence=COLORS2,
        facet_row="count_type",
        category_orders={"usertype": category_orders_usertype},
    )
    fig.update_traces(mode="markers+lines")
    fig.update_layout(hovermode="x unified", title_text="Trips per Day over Year")
    fig.update_xaxes(hoverformat="%a, %d %b %Y")
    fig.update_yaxes(matches=None, showticklabels=True)
    to_pickle(fig, "plot3")
    logger.info("Finished make_plot3")


# Plot 4: Trip Duration by Month
def make_plot4(df: pd.DataFrame):
    logger.info("Started make_plot4...")
    df["leq_3600"] = (df["trip_duration"] <= 3600).astype(int)
    df_plot4 = (
        df.groupby(["usertype", "leq_3600"])[["month"]]
        .agg(count=("month", "count"))
        .reset_index()
    )

    fig = px.bar(
        df_plot4,
        x="usertype",
        y="count",
        color="usertype",
        color_discrete_sequence=COLORS2,
        category_orders={
            "usertype": category_orders_usertype,
        },
        facet_row="leq_3600",
    )
    fig.update_yaxes(
        matches=None, showticklabels=True, title_text="Tripduration by usertype"
    )
    fig.update_layout(
        boxgap=0.125,
        boxgroupgap=0.125,
    )
    to_pickle(fig, "plot4")
    logger.info("Finished make_plot4")


# Plot 5: Trip Duration by usertype
def make_plot5(df: pd.DataFrame):
    logger.info("Started make_plot5...")

    df["leq_1h"] = (df["trip_duration_min"] <= 1).astype(int)
    df_plot5 = df.query("trip_duration <= 10800").sample(int(1e6))

    fig = px.histogram(
        df_plot5,
        x="trip_duration_min",
        color="usertype",
        color_discrete_sequence=COLORS2,
        category_orders={
            "month": category_orders_month,
            "usertype": category_orders_usertype,
        },
        facet_row="leq_1h",
        marginal="box",
        opacity=0.4,
        barmode="overlay",
    )
    fig.update_xaxes(matches=None, showticklabels=True)
    fig.update_yaxes(matches=None, showticklabels=True)
    fig.update_layout(hovermode="x unified", title_text="Tripduration by usertype")

    fig.layout.yaxis1.type = "log"
    to_pickle(fig, "plot5")
    logger.info("Finished make_plot5")


# Plot 6: Trip Duration by usertype
def make_plot6(df: pd.DataFrame):
    logger.info("Started make_plot6...")
    df_plot6 = df.query("trip_duration <= 7200").sample(int(1e6))

    fig = px.histogram(
        df_plot6,
        x="trip_duration_min",
        color="usertype",
        color_discrete_sequence=COLORS2,
        category_orders={
            "month": category_orders_month,
            "usertype": category_orders_usertype,
        },
        marginal="box",
        hover_data={
            "trip_duration_min": False,
            "usertype": False,
        },
        opacity=0.4,
        # nbins=200,
        histnorm="percent",  # "probability","density"
        barmode="overlay",
    )
    # fig.update_traces(xbins=dict(start=0.0, end=120, size=2))
    fig.update_layout(hovermode="x unified", title_text="Tripduration by usertype")
    to_pickle(fig, "plot6")
    logger.info("Finished make_plot6")


# Plot 7: birth year by usertype
def make_plot7(df: pd.DataFrame):
    logger.info("Started make_plot7...")
    df_plot7 = df.sample(int(1e6))

    fig = px.histogram(
        df_plot7,
        x="birth_year",
        color="usertype",
        color_discrete_sequence=COLORS2,
        marginal="box",
        hover_data={
            "usertype": False,
        },
        opacity=0.4,
        # nbins=200,
        histnorm="percent",  # "probability","density"
        barmode="overlay",
    )
    fig.update_xaxes(range=(1940, 2010))
    fig.update_layout(hovermode="x unified", title_text="Birthyear by usertype")
    to_pickle(fig, "plot7")
    logger.info("Finished make_plot7")


# Plot 8: gender by usertype
def make_plot8(df: pd.DataFrame):
    logger.info("Started make_plot8...")
    df_plot8 = (
        df.groupby(["usertype", "gender"])[["month"]]
        .agg(count=("month", "count"))
        .reset_index()
    )

    fig = px.bar(
        df_plot8,
        x="usertype",
        y="count",
        color="usertype",
        color_discrete_sequence=COLORS2,
        category_orders={
            "usertype": category_orders_usertype,
        },
        facet_row="gender",
    )
    fig.update_yaxes(matches=None, showticklabels=True, title_text="Trips by gender")
    fig.update_layout(
        boxgap=0.125,
        boxgroupgap=0.125,
    )
    to_pickle(fig, "plot8")
    logger.info("Finished make_plot8")


def make_locations_df(df: pd.DataFrame):
    df_station = pd.read_parquet("data/processed/df_station.parquet")
    df_plot9_start = (
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
    df_plot9_end = (
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
    df_plot9 = pd.concat([df_plot9_start, df_plot9_end]).drop_duplicates().dropna()
    df_plot9_loc = df_plot9.merge(
        df_station, how="left", on=["station_id", "station_name", "lat", "lng"]
    )
    df_plot9_loc["station_id"] = df_plot9_loc["station_id"].astype(int)
    df_plot9_loc = df_plot9_loc.filter(
        [
            "station_id",
            "station_name",
            "lat",
            "lng",
            "address",
            "suburb",
            "county",
            "city",
            "state",
            "neighbourhood",
            "postcode",
        ]
    )

    # df_plot9_loc = df_plot9_loc.query("city == 'City of New York'")
    # df_plot9_loc.loc[df_plot9_loc["suburb"].isna(), "suburb"] = "Manhattan"

    df_plot9_loc["suburb_missing"] = df_plot9_loc["suburb"].isna()
    df_plot9_loc["neighbourhood_missing"] = df_plot9_loc["neighbourhood"].isna()
    return df_plot9_loc


# Plot 9: Stations
def make_plot9(df):
    logger.info("Started make_plot9...")

    df_plot9 = make_locations_df(df)

    fig = go.Figure()
    for col, suburb in zip(COLORS4, df_plot9["suburb"].unique()):
        fig = fig.add_trace(
            go.Scattermapbox(
                name=suburb,
                lat=df_plot9.query(f"suburb == '{suburb}'")["lat"],
                lon=df_plot9.query(f"suburb == '{suburb}'")["lng"],
                marker=go.scattermapbox.Marker(
                    size=6,
                    color=col,
                    symbol="circle",
                ),
                text=df_plot9.query(f"suburb == '{suburb}'")["station_name"],
            )
        )
    fig.add_trace(
        go.Scattermapbox(
            name="Missing",
            lat=df_plot9.query(f"suburb_missing == 1")["lat"],
            lon=df_plot9.query(f"suburb_missing == 1")["lng"],
            marker=go.scattermapbox.Marker(
                size=6,
                color="gray",
                symbol="triangle",
            ),
            text=df_plot9.query(f"suburb_missing == 1")["station_name"],
        )
    )

    fig.update_layout(
        hovermode="closest",
        mapbox=dict(
            accesstoken=mapbox_access_token,
            center=go.layout.mapbox.Center(lat=40.759, lon=-73.988),
            zoom=11,
        ),
    )

    to_pickle(fig, "plot9")
    logger.info("Finished make_plot9")


def make_plot10(df: pd.DataFrame):
    logger.info("Started make_plot10...")

    df_plot10 = make_locations_df(df)
    N = df_plot10["neighbourhood"].nunique()
    colors = ["hsl(" + str(h) + ",50%" + ",50%)" for h in np.linspace(0, 360, N)]
    fig = go.Figure()
    for col, val in zip(colors, df_plot10["neighbourhood"].unique()):
        fig = fig.add_trace(
            go.Scattermapbox(
                name=val,
                lat=df_plot10.loc[df_plot10["neighbourhood"] == val, "lat"],
                lon=df_plot10.loc[df_plot10["neighbourhood"] == val, "lng"],
                marker=go.scattermapbox.Marker(
                    size=6,
                    color=col,
                    symbol="circle",
                ),
                text=df_plot10.loc[df_plot10["neighbourhood"] == val, "station_name"],
            )
        )
    fig.update_layout(
        hovermode="closest",
        mapbox=dict(
            accesstoken=mapbox_access_token,
            center=go.layout.mapbox.Center(lat=40.759, lon=-73.988),
            zoom=11,
        ),
    )
    fig.add_trace(
        go.Scattermapbox(
            name="Missing",
            lat=df_plot10.query(f"neighbourhood_missing == 1")["lat"],
            lon=df_plot10.query(f"neighbourhood_missing == 1")["lng"],
            marker=go.scattermapbox.Marker(
                size=6,
                color="gray",
                symbol="triangle",
            ),
            text=df_plot10.query(f"neighbourhood_missing == 1")["station_name"],
        )
    )

    to_pickle(fig, "plot10")
    logger.info("Started make_plot10...")


def make_plot11(df: pd.DataFrame):
    logger.info("Started make_plot11...")
    df = pd.read_parquet("data/processed/df_nyc.parquet")

    df_plot11 = (
        df.groupby(["month", "date", "usertype"])
        .agg(
            count=("start_station_id", "count"),
            temp=("temp", np.mean),
            temp_max=("temp_max", np.max),
            temp_min=("temp_max", np.max),
        )
        .reset_index()
    )
    df_plot11["month"] = df_plot11["month"].replace(
        {
            1: "Jan",
            2: "Feb",
            3: "Mar",
            4: "Apr",
            5: "May",
            6: "Jun",
            7: "Jul",
            8: "Aug",
            9: "Sep",
            10: "Oct",
            11: "Nov",
            12: "Dec",
        }
    )
    df_plot11 = df_plot11.rename(
        {"temp": "Avg. temp per Day", "count": "Trips per Day"}, axis=1
    )
    df_plot11["month x usertype"] = df_plot11["month"] + "_x_" + df_plot11["usertype"]
    fig = px.scatter(
        df_plot11,
        x="Avg. temp per Day",
        y="Trips per Day",
        color="month x usertype",
        symbol="usertype",
        facet_col="month",
        facet_col_wrap=4,
        category_orders={
            "month": category_orders_month,
            "usertype": category_orders_usertype,
        },
        color_discrete_sequence=COLORS12,
        facet_col_spacing=0.05,
        facet_row_spacing=0.05,
        trendline="lowess",
    )
    fig.update_xaxes(matches=None, showticklabels=True)
    fig.update_yaxes(matches=None, showticklabels=True)
    fig.update_layout(showlegend=False)
    # fig.show()
    to_pickle(fig, "plot11")
    logger.info("Started make_plot11...")


# Plot 12: Destinations
def make_plot12(df: pd.DataFrame):
    logger.info("Started make_plot12...")

    dfx = df.filter(
        [
            "starttime",
            "stoptime",
            "start_lat",
            "start_lng",
            "end_lat",
            "end_lng",
            "usertype",
        ]
    ).rename({"starttime": "start_time", "stoptime": "end_time"}, axis=1)
    df_plot12 = pd.concat(
        [
            dfx.filter(["start_time", "start_lat", "start_lng", "usertype"]).rename(
                {"start_time": "time", "start_lat": "lat", "start_lng": "lng"},
                axis=1,
            ),
            dfx.filter(["end_time", "end_lat", "end_lng", "usertype"]).rename(
                {"end_time": "time", "end_lat": "lat", "end_lng": "lng"}, axis=1
            ),
        ]
    )
    df_plot12["hour"] = df_plot12["time"].dt.hour

    fig = ff.create_hexbin_mapbox(
        data_frame=df_plot12,
        lat="lat",
        lon="lng",
        animation_frame="hour",
        nx_hexagon=50,
        opacity=0.5,
        labels={"color": "Point Count"},
        min_count=10,
        color_continuous_scale="Viridis",
        zoom=12,
    )
    to_pickle(fig, "plot12")
    logger.info("Finished make_plot12")


if __name__ == "__main__":

    logger.info("Started!")
    df = dd.read_parquet("data/interim/2018**.parquet").compute()
    df = df.query("start_lat != 0 and end_lng != 0 and start_lng != 0 and end_lng != 0")

    logger.info("Data loaded!")
    try:
        make_plot1(df)
    except:
        logger.warning("Couldn't crate plot 1")
    try:
        make_plot2(df)
    except:
        logger.warning("Couldn't crate plot 2")
    try:
        make_plot3(df)
    except:
        logger.warning("Couldn't crate plot 3")
    try:
        make_plot4(df)
    except:
        logger.warning("Couldn't crate plot 4")
    try:
        make_plot5(df)
    except:
        logger.warning("Couldn't crate plot 5")

    try:
        make_plot6(df)
    except Exception as e:
        logger.warning(f"Couldn't crate plot 6. Error: {e}")

    try:
        make_plot7(df)
    except Exception as e:
        logger.warning(f"Couldn't crate plot 7. Error: {e}")
    try:
        make_plot8(df)
    except Exception as e:
        logger.warning(f"Couldn't crate plot 8. Error: {e}")
    try:
        make_plot9(df)
    except Exception as e:
        logger.warning(f"Couldn't crate plot 9. Error: {e}")
    try:
        make_plot10(df)
    except Exception as e:
        logger.warning(f"Couldn't crate plot 10. Error: {e}")
    try:
        make_plot11(df)
    except Exception as e:
        logger.warning(f"Couldn't crate plot 11. Error: {e}")
    try:
        make_plot12(df)
    except Exception as e:
        logger.warning(f"Couldn't crate plot 12. Error: {e}")
