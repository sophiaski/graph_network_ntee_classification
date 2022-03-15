# Initialize directory paths, classes, constants, packages, and other methods
from utils import *


@typechecked
def load_gold_eins_with_labels(verbose: bool = True) -> pd.DataFrame:
    """_summary_

    Args:
        verbose (bool, optional): _description_. Defaults to True.

    Returns:
        pd.DataFrame: _description_
    """
    from baseline_silver_to_gold import create_gold

    # Read in benchmark data
    train, test = create_gold(include_ein=True)

    # Sequence data not needed in v1 GNN classifier
    df = pd.concat([train, test]).drop("sequence", axis=1)
    if verbose:
        print(f"Benchmark data: ({df.shape[0]:,}, {df.shape[1]:,})")
    return df


def create_graph():
    return None


def save_graph():
    return None


def main():
    """
    Create dataset for the model, saving to parquet for easy access.
    """
    return None


if __name__ == "__main__":
    main()
