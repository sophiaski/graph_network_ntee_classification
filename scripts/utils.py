"""Place where I store all of the functions that are useful in multiple scripts"""

# Constants needed to parse IRS 990s
from constants import *

# Custom class to define schema of row data
from classes import *

# Environment
import os

# Type hints
from typeguard import typechecked
from typing import Sequence, Union, Dict, Deque, Iterable, Mapping

# Data analysis
import pandas as pd
import numpy as np
import datetime
import time

# Get root directory and other directory paths to use in scripts
PROJECT_ROOT = os.path.dirname(os.path.abspath(os.curdir))
SCHEMA_PATH = f"{PROJECT_ROOT+'/schemas'}"
BRONZE_PATH = f"{PROJECT_ROOT+'/data/bronze'}"
SILVER_PATH = f"{PROJECT_ROOT+'/data/silver'}"
GOLD_PATH = f"{PROJECT_ROOT+'/data/gold'}"
for PATH in [SCHEMA_PATH, BRONZE_PATH, SILVER_PATH, GOLD_PATH]:
    if not os.path.exists(PATH):
        os.makedirs(PATH)


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
