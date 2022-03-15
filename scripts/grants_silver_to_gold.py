# Initialize directory paths, classes, constants, packages, and other methods
from utils import *


@typechecked
def load_silver_grants(
    frac: float = 1.0,
    seed: int = SEED,
    verbose: bool = True,
) -> pd.DataFrame:
    """Load in the silver grants data.

    Args:
        frac (float, optional): Return a random fraction of rows from Pandas DataFrame. Defaults to 1.0 (100%).
        seed (int, optional): Random state for reproducibiltiy. Defaults to SEED.
        verbose (bool, optional): Print Pandas DataFrame shape. Defaults to True.

    Returns:
        pd.DataFrame: Combined dataset.
    """
    filepath = f"{GRANTS_SILVER_PATH}grants.parquet"
    data = (
        pd.read_parquet(filepath)
        .sample(frac=frac, random_state=seed)
        .reset_index(drop=True)
    )

    if verbose:
        print(f"SAMPLING {round(frac*100,2)}% OF DATA...")
        print(f"\tDATA SIZE: {data.shape}")
    return data


@typechecked
def load_gold_eins_with_labels(verbose: bool = True) -> pd.DataFrame:
    """_summary_

    Args:
        verbose (bool, optional): _description_. Defaults to True.

    Returns:
        pd.DataFrame: _description_
    """
    from benchmark_silver_to_gold import create_gold

    # Read in benchmark data
    train, test = create_gold(include_ein=True)

    # Sequence data not needed in v1 GNN classifier
    df = pd.concat([train, test]).drop("sequence", axis=1)
    if verbose:
        print(f"Benchmark data: ({df.shape[0]:,}, {df.shape[1]:,})")
    return df


def create_graph():
    grants = load_silver_grants
    benchmark = load_gold_eins_with_labels()

    # Drop unnecesary columns (include edges)
    # Create nodes from grants, apply EINs from benchmark
    # Split into nodes and edges

    return None


def save_graph():
    # Save in networkx
    # Save in PyKeen
    return None


def main():
    """
    Create edges and nodes (gold) of the grants dataset, saving to parquet for easy access.
    """
    return None


if __name__ == "__main__":
    main()
