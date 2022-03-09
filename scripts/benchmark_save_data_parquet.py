# Initialize directory paths, classes, constants, packages, and other methods
from utils import *

HEADERS_BENCHMARK = ["input", "NTEE1", "broad_cat"]
SCHEMA_BENCHMARK = {val: pa.string() for val in HEADERS_BENCHMARK}

from collections import ChainMap

BROAD_CAT_MAPPER = dict(
    ChainMap(*[{letter: k for letter in v} for k, v in BROAD_CAT_DICT.items()])
)


@typechecked
def load_and_update_dataframes() -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Load in benchmark dataset and do some pre-pre-processing before incorporating into model.

    Returns:
        Tuple[pd.DataFrame, pd.DataFrame]: Output train and test dataframes.
    """
    train = pd.concat(
        [
            pd.read_pickle(f"{TRAIN_PATH}{file}", compression="gzip")
            for file in os.listdir(TRAIN_PATH)
        ],
        ignore_index=True,
    )

    test = pd.concat(
        [
            pd.read_pickle(f"{TEST_PATH}{file}", compression="gzip")
            for file in os.listdir(TEST_PATH)
        ],
        ignore_index=True,
    )
    for df in [train, test]:
        df["input"] = (
            df["TAXPAYER_NAME"]
            + " "
            + df["mission_spellchk"]
            + " "
            + df["prgrm_dsc_spellchk"]
        )

        df["broad_cat"] = df["NTEE1"].map(BROAD_CAT_MAPPER)
        df.drop(
            [c for c in df if c not in HEADERS_BENCHMARK], axis="columns", inplace=True
        )

    return train[HEADERS_BENCHMARK], test[HEADERS_BENCHMARK]


@typechecked
def save_to_parquet(data: pd.DataFrame, filename: str) -> None:
    """Save the pre-pre-processed dataframes into parquet files.

    Args:
        data (pd.DataFrame): Pandas DataFrame.
        filepath (str): Filename.
    """
    schema = pa.schema(SCHEMA_BENCHMARK)
    table = pa.Table.from_pandas(data, schema=schema)
    pq.write_table(
        table,
        where=f"{BENCHMARK_MODEL}df_ucf_{filename}.parquet",
        compression="snappy",
    )


def main():
    """
    Load and pre-process benchmark dataset, saving to parquet for easy access.
    """
    train, test = load_and_update_dataframes()
    save_to_parquet(data=train, filename="train")
    save_to_parquet(data=test, filename="test")


if __name__ == "__main__":
    main()
