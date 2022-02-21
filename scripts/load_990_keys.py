"""Load the IRS 990 filing keys from Amazon."""

### ADD CREDIT TO NOTEBOOK INSTRUCTIONS

# Iterating through Amazon IRS filings
from utils import *
import time
import datetime
from collections import deque
import logging
import boto3
from typing import Deque, Iterable
from concurrent.futures import (
    ProcessPoolExecutor,
    ThreadPoolExecutor,
    as_completed,
    Future,
)

# Save filing keys to CSV
import unicodecsv as csv

logging.basicConfig(
    format="%(asctime)s : %(levelname)s : %(message)s", level=logging.INFO
)


def get_keys_for_prefix(prefix: str) -> Iterable[str]:
    """Return a collection of all key names starting with the specified prefix."""
    session = boto3.session.Session()
    client = session.client("s3")

    # See https://boto3.amazonaws.com/v1/documentation/api/latest/guide/paginators.html
    paginator = client.get_paginator("list_objects_v2")
    page_iterator = paginator.paginate(Bucket=BUCKET, Prefix=prefix)

    # A deque is a collection with O(1) appends and O(n) iteration
    results: Deque[str] = deque()
    i = 0
    for i, page in enumerate(page_iterator):
        if "Contents" not in page:
            continue

        # You could also capture, e.g., the timestamp or checksum here
        page_keys: Iterable = (element["Key"] for element in page["Contents"])
        results.extend(page_keys)
    logging.info("Scanned {} page(s) with prefix {}.".format(i + 1, prefix))
    return results


for YEAR in range(2009, datetime.datetime.now().year + 1):

    logging.info(f"Starting year {YEAR}")
    BUCKET: str = "irs-form-990"
    EARLIEST_YEAR = YEAR
    cur_year = YEAR
    first_prefix: int = EARLIEST_YEAR * 100
    last_prefix: int = (cur_year + 1) * 100

    start: float = time.time()

    # ProcessPoolExecutor starts a completely separate copy of Python for each worker
    with ProcessPoolExecutor() as executor:
        futures: Deque[Future] = deque()
        for prefix in range(first_prefix, last_prefix):
            future: Future = executor.submit(get_keys_for_prefix, str(prefix))
            futures.append(future)

    # the name of the output file
    outfile = open(f"{BRONZE_PATH}/990_filing_keys/filing_numbers_{YEAR}.csv", "wb")
    # start up a dictwriter, ignore extra rows
    dw = csv.DictWriter(outfile, ["KEY"], extrasaction="ignore")
    dw.writeheader()

    n = 0
    for future in as_completed(futures):
        keys: Iterable = future.result()
        for key in keys:
            dw.writerow({"KEY": int(key.replace("_public.xml", ""))})
            n += 1

    elapsed: float = time.time() - start

    logging.info("Discovered {:,} keys in {:,.1f} seconds.".format(n, elapsed))
