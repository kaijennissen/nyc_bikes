import glob
import logging

import coloredlogs
import pandas as pd

logger = logging.getLogger("helpers")
coloredlogs.install(
    fmt="%(asctime)s - %(name)s - %(process)s - %(levelname)s - %(message)s",
    level="DEBUG",
)


def save_to_dir(df: pd.DataFrame, outdir: str = "data/processed"):
    grouped = df.groupby(["year", "month"])
    for (year, month), df_m in grouped:
        outfile = outdir + f"/{year}{str(month).zfill(2)}.parquet"
        logger.debug(f"writing to {outfile}")
        df_m.to_parquet(outfile)


def read_from_dir(indir: str = "data/processed"):
    ls_ = []
    files = glob.glob(indir + "/2018*.parquet")
    for file_ in files:
        logger.debug(f"reading from {file_}")
        ls_.append(pd.read_parquet(file_))
    df = pd.concat(ls_)
    logger.debug("Combined all files!")
    return df
