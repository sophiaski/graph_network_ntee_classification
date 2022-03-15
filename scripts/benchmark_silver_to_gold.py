# Initialize directory paths, classes, constants, packages, and other methods
from utils import *


@typechecked
def create_gold(
    include_ein: bool = False,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Load in benchmark dataset and do some additional processing before incorporating into model.

    Returns:
        Tuple[pd.DataFrame, pd.DataFrame]: Gold train and test dataframes.
    """

    # Get mapper that converts from NTEE1 to Broad category
    from collections import ChainMap

    # Get constants
    BROAD_CAT_MAPPER = dict(
        ChainMap(*[{letter: k for letter in v} for k, v in BROAD_CAT_DICT.items()])
    )
    TRAIN_PATH = f"{BENCHMARK_SILVER_PATH+'train/'}"
    TEST_PATH = f"{BENCHMARK_SILVER_PATH+'test/'}"

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
        df.loc[:, "sequence"] = (
            df["TAXPAYER_NAME"]
            + " "
            + df["mission_spellchk"]
            + " "
            + df["prgrm_dsc_spellchk"]
        )

        df.loc[:, "broad_cat"] = df["NTEE1"].map(BROAD_CAT_MAPPER)

    if not include_ein:
        return train[BENCHMARK_HEADERS], test[BENCHMARK_HEADERS]
    else:
        return train[["EIN"] + BENCHMARK_HEADERS], test[["EIN"] + BENCHMARK_HEADERS]


def main():
    """
    Create gold (model) dataset, saving to parquet for easy access.
    """
    train, test = create_gold(include_ein=False)
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


if __name__ == "__main__":
    main()
