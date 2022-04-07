# Initialize directory paths, classes, constants, packages, and other methods
from utils import *


@typechecked
def create_benchmark() -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Load in benchmark dataset and do some additional processing before incorporating into model.

    Returns:
        Tuple[pd.DataFrame, pd.DataFrame]: Train and test DataFrames.
    """

    # Get mapper that converts from NTEE1 to Broad category
    from collections import ChainMap

    # Get constants
    BROAD_CAT_MAPPER = dict(
        ChainMap(*[{letter: k for letter in v} for k, v in BROAD_CAT_DICT.items()])
    )
    TRAIN_PATH = f"{BENCHMARK_SILVER_PATH}train/"
    TEST_PATH = f"{BENCHMARK_SILVER_PATH}test/"

    # Load in pickle files from benchmark.
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

    # Create sequence column
    for df in [train, test]:
        df.loc[:, "sequence"] = (
            df["TAXPAYER_NAME"]
            + " "
            + df["mission_spellchk"]
            + " "
            + df["prgrm_dsc_spellchk"]
        )

        df.loc[:, "broad_cat"] = df["NTEE1"].map(BROAD_CAT_MAPPER)

    # Make sure EINs are of length 9 by padding with '0' from the left.
    for df in [train, test]:
        df.loc[:, "EIN"] = (
            df["EIN"].astype(str).apply(lambda val: f"{'0'*(9-len(val))}{val}")
        )
        df.rename(
            {
                "EIN": "ein",
                "TAXPAYER_NAME": "taxpayer_name",
                "RETURN_TYPE": "return_type",
            },
            axis=1,
            inplace=True,
        )

    # Depending on use case, include the EIN as another column.
    return train[BENCHMARK_HEADERS], test[BENCHMARK_HEADERS]


@typechecked
def load_benchmark(
    merge_data: bool = True, frac: float = 1.0, seed: int = SEED, verbose: bool = True,
) -> Union[Tuple[pd.DataFrame, pd.DataFrame], pd.DataFrame]:
    """Load in the cleaned grants data.

    Args:
        merge_data (bool, optional): If True, merge the train/test DataFrames. Defaults to True.
        frac (float, optional): Return a random fraction of rows from Pandas DataFrame. Defaults to 1.0 (100%).
        seed (int, optional): Random state for reproducibiltiy. Defaults to SEED.
        verbose (bool, optional): If True, print Pandas DataFrame shape. Defaults to True.

    Returns:
        pd.DataFrame: Gold grants DataFrame(s) -> combined or (train, test)
    """
    loc = BENCHMARK_GOLD_PATH

    train = load_parquet(
        loc=loc, filename="df_ucf_train", frac=frac, seed=seed, verbose=verbose
    )
    test = load_parquet(
        loc=loc, filename="df_ucf_test", frac=frac, seed=seed, verbose=verbose
    )
    if merge_data:
        train.loc[:, "benchmark_status"] = "train"
        test.loc[:, "benchmark_status"] = "test"
        return (
            pd.concat([train, test])
            .sample(frac=frac, random_state=seed)
            .reset_index(drop=True)
        )
    return train, test


def main():
    """
    Create gold (model) dataset, saving to parquet for easy access.
    """
    train, test = create_benchmark()
    save_to_parquet(
        data=train,
        cols=BENCHMARK_HEADERS,
        loc=BENCHMARK_GOLD_PATH,
        filename="df_ucf_train",
    )
    save_to_parquet(
        data=test,
        cols=BENCHMARK_HEADERS,
        loc=BENCHMARK_GOLD_PATH,
        filename="df_ucf_test",
    )
    return train, test


if __name__ == "__main__":
    main()
