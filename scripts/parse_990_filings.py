"""Iterate through the filing keys and extract text fields using IRSx parser"""

### ADD CREDIT TO IRS COOKBOOK

# Started on Feb 20th
# Ended on TBD

# Initialize directory paths, classes, constants, packages, and other methods
from utils import *

# IRS parsing
from irsx.xmlrunner import XMLRunner

# Data analysis, writing to parquet
import pyarrow as pa
import pyarrow.parquet as pq

# Load data capture dictionary
import json

# Rapids 0.19 Environment
import cudf


@typechecked
def batch_save_to_parquet(
    year: int,
    rows: Sequence[RowData],
    partition_no: int = 999,
    verbose: bool = True,
) -> None:
    """Save every 10,000 parsed IRS filings to a parquet file.

    Args:
        year (int): Year the 990 was filed.
        rows (Sequence[RowData]): Parsed row data.
        partition_no (int): The value we'll use to indicate the partition number. Defaults to 999.
        verbose (bool, optional): Print number of parquet batches saved for debugging. Defaults to True.
    """
    schema = pa.schema(SCHEMA_990)
    data = pd.DataFrame(data=rows, columns=HEADERS_990).fillna(
        "NaN"
    )  # Schema requires that all columns are strings
    table = pa.Table.from_pandas(data=data, schema=schema)
    pq.write_table(
        table,
        where=f"{BRONZE_PATH}/990_filings_parsed/parsed_990s_{year-1}_{partition_no}.parquet",
        compression="snappy",
    )
    if verbose:
        print(f"Batch save no. {partition_no}")


@typechecked
def run_filing(key: np.int64, verbose: bool = False) -> RowData:
    """Extract the mission and program descriptions from inividual IRS XML files.

    Args:
        key (numpy.int64): 990 filing document number.
        verbose (bool, optional): Debugging print statements. Defaults to False.

    Returns:
        RowData: TypedDict from 'classes.py' that specifies the schema of the output dictionary. If the filing is not able to be parsed, return row with just the object_id.
    """

    if verbose:
        print(f"FILING NO. {key}")
    try:
        parsed_filing = xml_runner.run_filing(key).get_result()
        if not parsed_filing:
            if verbose:
                print(
                    f"SKIPPING FILING NO. {key} (filings with pre-2013 filings are skipped)"
                )
            return {"object_id": key}
    except:
        return {"object_id": key}

    # Get list of schedules for the specific filing
    locator = {}
    for idx, sked in enumerate(parsed_filing):
        locator[sked["schedule_name"]] = idx

    # Store the data for the new csv output file here
    row_data = {}

    # If the filing we want was actually found by the XML runner, parse it
    for sked in set(DATA_CAPTURE_DICT.keys()) & set(locator.keys()):

        if verbose:
            print(f"\tSchedule {sked}")

        parsed_sked = parsed_filing[locator[sked]]

        # If this particular filing has group fields to extract
        if WANT_SKED_GROUPS[sked]:

            # Use schema dict to see if it has the groups we want
            for group in DATA_CAPTURE_DICT[sked]["groups"]:

                if verbose:
                    print(f"\t\tRepeating group {group}")

                # Parse chunk
                groups = parsed_sked["groups"].get(group)
                if groups:
                    if verbose:
                        print(f"\t\t\tFound {len(groups)} groups for {group}")
                else:
                    if verbose:
                        print(f"\t\t\tNo groups found for {group}")
                    continue

                # Use schema dict to get variable name to search each chunk for
                capture_dict = DATA_CAPTURE_DICT[sked]["groups"][group]
                variable_name = list(capture_dict.keys())[0]
                csv_header = capture_dict[variable_name]["header"]
                if verbose:
                    print(f"\t\t\t\t{variable_name} -> {csv_header}")

                # For each group chunk, append to create one long string variable
                grouped_data = []
                for parsed_group in groups:
                    val = parsed_group.get(variable_name)
                    if val:
                        grouped_data.append(val)
                    if verbose:
                        print(f"\t\t\t\t\t{val}")

                # If the array isn't empty after looping through the groups, then join by unique character
                if grouped_data:
                    row_data[csv_header] = "|".join(grouped_data)

        # If this particular filing has schedule parts fields to extract
        if WANT_SKED_PARTS[sked]:

            # Create smaller data dictionary to segment by schedule sub-parts
            capture_dict = DATA_CAPTURE_DICT[sked]["schedule_parts"]
            for sub_part in capture_dict.keys():

                # Make sure the sub-part is in the parsed data
                sub_part_parsed = parsed_sked["schedule_parts"].get(sub_part)
                if not sub_part_parsed:
                    continue
                if verbose:
                    print(f"\t\t{sub_part}")

                # Iterate across the keys in the sub-part to capture value and append to row data
                for variable_name in capture_dict[sub_part].keys():
                    if variable_name in sub_part_parsed.keys():
                        csv_header = capture_dict[sub_part][variable_name]["header"]
                        val = sub_part_parsed[variable_name]
                        row_data[csv_header] = val
                        if verbose:
                            print(f"\t\t\t{variable_name} -> {csv_header}")
                            print(f"\t\t\t\t{val}")
    return row_data


###########################################################
###########################################################
###########################################################
###########################################################

# Load data capture dictionary
DATA_CAPTURE_DICT = json.load(open(SCHEMA_PATH + "/xml_data_capture.json"))

# Create simple schema
SCHEMA_990 = {val: pa.string() for val in HEADERS_990}
SCHEMA_990["object_id"] = pa.int64()

# What size chunks to partition parquet files by
cycle = 10000

# Where are we starting the
start_year = 2014
end_year = 2014
ticker, start_here = 0, 0

for year in range(start_year, end_year + 1):
    xml_runner = XMLRunner()

    # Import IRS key list saved from Amazon
    key_list = (
        cudf.read_csv(f"{BRONZE_PATH}/990_filing_keys/filing_numbers_{year}.csv")
        .KEY.to_arrow()
        .to_numpy()
    )
    # For each key, parse the XML file as outlined in the data dictionary schema, save to parquet
    ticker, start_here = 80001, 80001  # In case we need to start between cycles
    rows = []
    for key in key_list[start_here:]:
        ticker += 1
        # Do your analysis here
        new_row = run_filing(key=key)
        if new_row:
            rows.append(new_row)
        if ticker % 10 == 0:
            print(ticker, end=" ")  # Only print every 10 records parsed.
        # Batch save to parquet
        if ticker % cycle == 0:
            batch_save_to_parquet(
                year=year,
                rows=rows,
                partition_no=int(ticker / cycle),
            )
            rows = []
    # Get the rest of the rows, save as "999"th cycle
    batch_save_to_parquet(year=year, ticker=ticker, rows=rows)
