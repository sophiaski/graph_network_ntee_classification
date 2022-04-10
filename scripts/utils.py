"""Place where I store all of the functions that are useful in multiple scripts"""

# Constants needed to parse IRS 990s
from constants import *

# Custom class to define schema of row data
from classes import *

# Environment
import os

# Type hints
from typeguard import typechecked
from typing import (
    Sequence,
    Union,
    Dict,
    Deque,
    Iterable,
    Mapping,
    Tuple,
    List,
    Optional,
    Any,
)

# Data analysis
import pandas as pd
import numpy as np
import datetime
import time
from collections import Counter
import random

# Graph time!
import networkx as nx

# Writing to parquet
import pyarrow as pa
import pyarrow.parquet as pq

# Neural network
import torch

# Progress bars
from tqdm.notebook import tqdm

# WandB
import wandb

# Date
from datetime import date

# Splitting data
from sklearn.model_selection import train_test_split

# Preparing data
from sklearn import preprocessing

# Ignore excessive warnings
import logging

logging.propagate = False
logging.getLogger().setLevel(logging.ERROR)

# Get root directory and other directory paths to use in scripts
PROJECT_ROOT = os.path.dirname(os.path.abspath(os.curdir))
SCHEMA_PATH = f"{PROJECT_ROOT+'/schemas/'}"
MODELS_PATH = f"{PROJECT_ROOT+'/models/'}"
EMBEDDINGS_PATH = f"{PROJECT_ROOT+'/embs/'}"

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

# For loading and saving the graph data
GRAPH_GOLD_PATH = f"{GOLD_PATH+'graph/'}"


@typechecked
def Merge(dict1: Mapping, dict2: Mapping) -> None:
    """Merge two dictionaries"""
    return dict1.update(dict2)


@typechecked
def save_to_parquet(
    data: pd.DataFrame, cols: Sequence[str], loc: str, filename: str
) -> None:
    """Save the processed dataframes into parquet files.

    Args:
        data (pd.DataFrame): Input Pandas DataFrame.
        cols (Sequence[str]): Column names.
        loc (str): Folder location.
        filename (str): Filename.
    """
    schema = pa.schema({val: pa.string() for val in cols})
    table = pa.Table.from_pandas(data, schema=schema)
    pq.write_table(
        table, where=f"{loc}{filename}.parquet", compression="snappy",
    )


@typechecked
def load_parquet(
    loc: str, filename: str, frac: float = 1.0, seed: int = SEED, verbose: bool = True,
) -> pd.DataFrame:
    """Load in the cleaned grants data.

    Args:
        frac (float, optional): Return a random fraction of rows from Pandas DataFrame. Defaults to 1.0 (100%).
        seed (int, optional): Random state for reproducibiltiy. Defaults to SEED.
        verbose (bool, optional): Print Pandas DataFrame shape. Defaults to True.

    Returns:
        pd.DataFrame: Gold grants DataFrame.
    """
    # Load in data
    filepath = f"{loc}{filename}.parquet"

    df = (
        pd.read_parquet(filepath)
        .sample(frac=frac, random_state=seed)
        .reset_index(drop=True)
        .replace("", pd.NA)
    )

    # Check it out
    if verbose:
        print(f"Sampling {round(frac*100,2)}% of {filename.upper()} data...")
        print(f"\tShape: {df.shape}")

    # Return dataframe with columns sorted alphabetically
    return df[sorted(df.columns)]


@typechecked
def load_json(loc: str, filename: str) -> Dict:
    """Load a json file as a python dictionary.

    Args:
        loc (str): Save location.
        filename (str): Name of file.

    Returns:
        Dict[str, str]: Dictionary that connects EINs to 990 fields {ein : field}
    """
    import json

    # Opening JSON file
    with open(f"{loc}{filename}.json") as infile:
        return json.load(infile)


@typechecked
def save_to_json(data: Mapping, loc: str, filename: str) -> None:
    """Save a dictionary to json.

    Args:
        data (Mapping): Input dictionary file.
        loc (str): Save location.
        filename (str): Name of file.
    """
    import json

    with open(f"{loc}{filename}.json", "w") as outfile:
        json.dump(data, outfile)


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


def readable_list(seq: List[Any]) -> str:
    """Return a grammatically correct human readable string (with an Oxford comma)."""
    # Ref: https://stackoverflow.com/a/53981846/
    seq = [str(s) for s in seq]
    if len(seq) < 3:
        return " and ".join(seq)
    return ", ".join(seq[:-1]) + ", and " + seq[-1]


# from prettytable import PrettyTable

# def count_parameters(model: torch.Module):
#     table = PrettyTable(["Modules", "Parameters"])
#     total_params = 0
#     for name, parameter in model.named_parameters():
#         if not parameter.requires_grad:
#             continue
#         params = parameter.numel()
#         table.add_row([name, params])
#         total_params += params
#     print(table)
#     print(f"Total Trainable Params: {total_params}")
#     return total_params
