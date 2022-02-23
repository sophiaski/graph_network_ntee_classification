# Initialize directory paths, classes, constants, packages, and other methods
from utils import *

# Save filing keys to CSV
import unicodecsv as csv

# Iterating through Amazon IRS filings
from collections import deque
import logging
import boto3
from concurrent.futures import (
    ProcessPoolExecutor,
    ThreadPoolExecutor,
    as_completed,
    Future,
)


@typechecked
def get_keys_for_prefix(prefix: str, BUCKET: str) -> Iterable[str]:
    """Return a collection of all key names starting with the specified prefix.

    Args:
        prefix (str): Prefix from AWS.

    Returns:
        Iterable[str]: Key location of XML file.
    """
    session = boto3.session.Session()
    client = session.client("s3")

    # See https://boto3.amazonaws.com/v1/documentation/api/latest/guide/paginators.html
    paginator = client.get_paginator("list_objects_v2")
    page_iterator = paginator.paginate(Bucket="irs-form-990", Prefix=prefix)

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


def main():
    """
    Load the IRS 990 Filings keys from Registry of Open Data on AWS.

    I tweaked a script from Applied Nonprofit Research to obtain the listings
    for  all filings found in the dataset.

    Article: https://appliednonprofitresearch.com/posts/2020/06/skip-the-irs-990-efile-indices/
    """
    logging.basicConfig(
        format="%(asctime)s : %(levelname)s : %(message)s", level=logging.INFO
    )
    for YEAR in range(2009, datetime.datetime.now().year + 1):

        logging.info(f"Starting year {YEAR}")
        EARLIEST_YEAR: int = YEAR
        cur_year: int = YEAR
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


if __name__ == "__main__":
    main()
