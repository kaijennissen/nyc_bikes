import logging

import coloredlogs
import pandas as pd

if __name__ == "__main__":
    logger = logging.getLogger("get_weather")
    coloredlogs.install(
        fmt="%(asctime)s - %(name)s - %(process)s - %(levelname)s - %(message)s",
        level="DEBUG",
    )
    logger.info("Starting get_weather.")
    df_weather = pd.read_csv(
        "data/external/weather_history.csv", parse_dates=["dt_iso"]
    )
    df_weather = df_weather.query("dt_iso >= '2017-12-30' and dt_iso <= '2019-01-02'")
    df_weather["date"] = pd.to_datetime(
        df_weather.dt_iso.str[:-9] + "UTC", format="%Y-%m-%d %H:%M:%S %Z"
    ).dt.tz_convert("US/Eastern")
    df_weather["date"] = df_weather.date.dt.tz_localize(None)
    df_weather.query("date >= '2018-01-01' and date <= '2018-12-31'")
    cols = df_weather.columns.tolist()
    df_weather = df_weather.filter(["date"] + cols[3:-1])
    df_weather.reset_index().to_parquet("data/external/weather_history_2018.parquet")
    logger.info("Finished get_weather.")
