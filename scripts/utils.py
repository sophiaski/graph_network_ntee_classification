"""Place where I store all of the functions that are useful in multiple scripts"""

# Constants needed to parse IRS 990s
from constants import *

# Custom class to define schema of row data
from classes import *

# Environment
import os

# Type hints
from typeguard import typechecked
from typing import Sequence, Union, Dict, Deque, Iterable, Mapping, Tuple, List

# Data analysis
import pandas as pd
import numpy as np
import datetime
import time
from collections import Counter
import random

# Writing to parquet
import pyarrow as pa
import pyarrow.parquet as pq

# Neural network
import torch

# Progress bars
from tqdm.notebook import tqdm

# WandB
import wandb

# Ignore excessive warnings
import logging

logging.propagate = False
logging.getLogger().setLevel(logging.ERROR)

# Get root directory and other directory paths to use in scripts
PROJECT_ROOT = os.path.dirname(os.path.abspath(os.curdir))
SCHEMA_PATH = f"{PROJECT_ROOT+'/schemas/'}"
MODELS_PATH = f"{PROJECT_ROOT+'/models/'}"

# For saving intermediate dataset processing
BRONZE_PATH = f"{PROJECT_ROOT+'/data/bronze/'}"
SILVER_PATH = f"{PROJECT_ROOT+'/data/silver/'}"
GOLD_PATH = f"{PROJECT_ROOT+'/data/gold/'}"

# For loading and saving the benchmark dataset
BENCHMARK_SILVER_PATH = f"{SILVER_PATH+'benchmark/'}"
BENCHMARK_GOLD_PATH = f"{GOLD_PATH+'benchmark/'}"

# For loading and saving the grants dataset
GRANTS_SILVER_PATH = f"{SILVER_PATH+'grants/'}"
GRANTS_GOLD_PATH = f"{GOLD_PATH+'grants/'}"


@typechecked
def save_to_parquet(
    data: pd.DataFrame, cols: Sequence[str], loc: str, filename: str
) -> None:
    """Save the pre-processed dataframes into parquet files.

    Args:
        data (pd.DataFrame): Input Pandas DataFrame.
        cols (Sequence[str]): Column names.
        loc (str): Folder location.
        filename (str): Filename.
    """
    schema = pa.schema({val: pa.string() for val in cols})
    table = pa.Table.from_pandas(data, schema=schema)
    pq.write_table(
        table,
        where=f"{loc}{filename}.parquet",
        compression="snappy",
    )


@typechecked
def import_or_install(package: str) -> None:
    """
    This code simply attempt to import a package, and if it is unable to, calls pip and attempts to install it from there.
    From: https://stackoverflow.com/questions/4527554/check-if-module-exists-if-not-install-it

    Args:
        package (str): python package name
    """
    import pip

    try:
        __import__(package)
    except ImportError:
        pip.main(["install", package])


@typechecked
def format_time(elapsed: float) -> str:
    """
    Takes a time in seconds and returns a string hh:mm:ss
    """
    # Round to the nearest second.
    elapsed_rounded = int(round((elapsed)))

    # Format as hh:mm:ss
    return str(datetime.timedelta(seconds=elapsed_rounded))
