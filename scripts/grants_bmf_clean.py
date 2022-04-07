# Initialize directory paths, classes, constants, packages, and other methods
from utils import *


@typechecked
def get_bmf_data(col: str, filler: str, verbose: bool = True) -> Dict[str, str]:
    """Create {EIN : col} dictionary for additional data clean-up.

    Args:
        col (str): Column to extract from the BMF CSVs.
        filler (str): When values are missing, fill with this value.
        verbose (bool, optional): Print statements for debugging. Defaults to True.

    Returns:
        Dict[str,str]: Data clean-up dictionary.
    """
    ein2col, bmf_dir = {}, f"{BRONZE_PATH}bmf/"

    # For each file in the bmf directory (sorted from earlier -> latest), extract EIN / col and convert to dictionary
    for file in sorted(os.listdir(bmf_dir)):
        if verbose:
            print(file)
        file_dict = (
            pd.read_csv(
                f"{bmf_dir}{file}",
                usecols=["EIN", col],
                dtype={"EIN": "str", col: "str"},
            )
            .drop_duplicates()
            .set_index("EIN")[col]
            .fillna(filler)
            .to_dict()
        )
        # Create one giant dictionary, if the same key is found, update it with the more recent value
        Merge(ein2col, file_dict)

    return ein2col


@typechecked
def get_propublica_data(
    eins: Sequence[str], col: str, how_long: int, verbose: bool = True
) -> Tuple[Dict[str, str], List[str]]:
    """Create {EIN : col} dictionary for additional data clean-up, and [ein1, ein2, ein3] list of EINs for debugging.

    Args:
        eins (Sequence[str]): Input sequence of EINs.
        col (str): Column to extract from ProPublica Nonprofit API.
        how_long (int): Length of string to extract from output.
        verbose (bool, optional): Print statements for debugging. Defaults to True.

    Returns:
        Tuple[Dict[str,str], List[str]]: Data clean-up dictionary and list.
    """
    from nonprofit import Nonprofit
    import time

    np = Nonprofit()
    new_vals, not_found = {}, []
    if verbose:
        print(f"Iterating over {len(eins):,} {col.upper()}s")
    # Get org information for each EIN in the input sequence list
    for idx, ein in enumerate(eins):
        if verbose:
            print(idx, end=" ")
        try:
            new_vals[ein] = np.orgs.get(ein)[col][:how_long]
            time.sleep(0.05)
        except:
            not_found.append(ein)
            continue
    return new_vals, not_found


def main():
    """
    See if you can find missing code and location fields using BMF and ProPublica API.
    """
    from w266.graph_network_ntee_classification.scripts.grants_process import prep_nodes

    nodes = prep_nodes(frac=1.0, seed=SEED, verbose=False)
    col_lst = [
        ("ZIP5", "zipcode", "zipcode", "00000", 5),
        ("CITY", "city", "city", "Unknown", 100),
        ("STATE", "state", "state", "ZZ", 2),
        ("NTEE1", "ntee_code", "NTEE1", "Z", 1),
        ("NAME", "name", "org", "Unknown", 100),
    ]
    for col1, col2, col3, filler, how_long in col_lst:

        #############################
        ### Round 1, use BMF CSVs ###
        #############################

        ein2col = get_bmf_data(col=col1, filler=filler, verbose=True)
        nodes.loc[nodes[col3] == filler, col3] = (
            nodes[nodes[col3] == filler]["ein"].map(ein2col).fillna(filler)
        )
        eins = nodes.loc[nodes[col3] == filler]["ein"].unique().tolist()
        print(f"# Missing after filling in w/ BMF data: {len(eins)}")

        ########################
        ### Round 2, use API ###
        ########################

        ein2col2, _ = get_propublica_data(
            eins=eins, col=col2, how_long=how_long, verbose=True
        )
        # Create one giant dictionary, if the same key is found, update it with the more recent value.
        Merge(ein2col, ein2col2)
        nodes.loc[nodes[col3] == filler, col3] = (
            nodes[nodes[col3] == filler]["ein"].map(ein2col).fillna(filler)
        )
        eins = nodes.loc[nodes[col3] == filler]["ein"].unique().tolist()
        print(f"# Missing after filling in w/ ProPublica data: {len(eins)}")

        # Save to json
        cleanup_dir = f"{BRONZE_PATH}cleanup/"
        print(f"Saving to {cleanup_dir}")
        save_to_json(data=ein2col, loc=cleanup_dir, filename=f"ein_{col1}")


if __name__ == "__main__":
    main()
