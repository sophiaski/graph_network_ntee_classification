# Initialize directory paths, classes, constants, packages, and other methods
from utils import *

# Get other methods to load in cleaned data
from grants_process import load_grants
from benchmark_process import load_benchmark


@typechecked
def create_nodes(
    frac: float = 1.0, seed: int = SEED, verbose: bool = True,
) -> pd.DataFrame:
    """Create the nodes DataFrame.

    Args:
        frac (float, optional): Return a random fraction of rows from Pandas DataFrame. Defaults to 1.0 (100%).
        seed (int, optional): Random state for reproducibiltiy. Defaults to SEED.
        verbose (bool, optional): Print Pandas DataFrame shape. Defaults to True.
        drop_edge_data (bool, optional): Drop edge features. Defaults to True.
        drop_xtra_node_data (bool, optional): Drop node features, execept for ones that are useful in building graph or understanding missing data. Defaults to True.
        drop_seq_data (bool, optional): Drop sequence data. Defaults to True.

    Returns:
        pd.DataFrame: Node DataFrame.
    """
    # Get mapper that converts from NTEE1 to Broad category
    from collections import ChainMap, OrderedDict

    grants = load_grants(frac=frac, seed=seed, verbose=verbose)
    benchmark = load_benchmark(merge_data=True, frac=frac, seed=seed, verbose=verbose)

    # Create column renamer!
    rename_mapper = {}
    for org_type in ["grantee", "grantor"]:
        rename_mapper[org_type] = "org"
        for col in ["ein", "city", "state", "zipcode", "location", "info"]:
            rename_mapper[f"{org_type}_{col}"] = col

    # Extract grantee, grantor columns, we will stack the separate DFs on top of each other and de-dupe
    grantee_cols = [field for field in grants.columns.tolist() if "grantee" in field]
    grantor_cols = [field for field in grants.columns.tolist() if "grantor" in field]

    # Apply column renamer
    grantees = grants[grantee_cols].rename(rename_mapper, axis=1)
    grantors = grants[grantor_cols].rename(rename_mapper, axis=1)

    # Add node type
    grantees.loc[:, "node_type"] = "grantee"
    grantors.loc[:, "node_type"] = "grantor"

    # Concatenate DataFrames
    grantees_and_grantors = pd.concat([grantees, grantors])

    # Merge DFs: Apply target labels to (where possible) org EINs
    nodes = (
        pd.merge(grantees_and_grantors, benchmark, how="outer", on="ein")
        .reset_index(drop=True)
        .fillna(pd.NA)
    )

    # Fill in missing org name values w/ taxpayer_name values
    nodes.loc[nodes["org"].isna() & nodes["taxpayer_name"].notna(), "org"] = nodes[
        nodes["org"].isna() & nodes["taxpayer_name"].notna()
    ]["taxpayer_name"]

    # Fill in missing sequence values w/ org name values
    nodes.loc[nodes["sequence"].isna() & nodes["org"].notna(), "sequence"] = nodes[
        nodes["sequence"].isna() & nodes["org"].notna()
    ]["org"]

    # Filling in missing sequences with grant_desc field from grants
    grant_desc_mapper = (
        grants.loc[
            grants["grantee_ein"].isin(nodes[nodes["sequence"].isna()]["ein"].tolist()),
            ["grantee_ein", "grant_desc"],
        ]
        .sort_values("grant_desc")
        .drop_duplicates(subset="grantee_ein")
        .set_index("grantee_ein")["grant_desc"]
        .to_dict()
    )
    nodes.loc[nodes["sequence"].isna(), "sequence"] = nodes[nodes["sequence"].isna()][
        "ein"
    ].map(grant_desc_mapper)

    # See if we can find more NTEE1 labels
    mapper = load_json(loc=f"{BRONZE_PATH}cleanup/", filename=f"ein_NTEE1")
    nodes.loc[nodes["NTEE1"].isna(), "NTEE1"] = (
        nodes[nodes["NTEE1"].isna()]["ein"].map(mapper).fillna("Z")
    )
    nodes.loc[:, "NTEE1"] = nodes[
        "NTEE1"
    ].str.upper()  # Some of the values are lowercase?

    # Get constants
    BROAD_CAT_MAPPER = dict(
        ChainMap(*[{letter: k for letter in v} for k, v in BROAD_CAT_DICT.items()])
    )
    nodes.loc[nodes["broad_cat"].isna(), "broad_cat"] = nodes[
        nodes["broad_cat"].isna()
    ]["NTEE1"].map(BROAD_CAT_MAPPER)

    # Create node_type feature
    missing_node_type = set(nodes[nodes["node_type"].isna()]["ein"].unique())
    grantee_node_type = set(nodes[nodes["node_type"] == "grantee"]["ein"].unique())
    grantor_node_type = set(nodes[nodes["node_type"] == "grantor"]["ein"].unique())
    both_node_type = grantee_node_type.intersection(grantor_node_type)

    nodes.loc[
        (nodes["NTEE1"] == "T") & (nodes["node_type"].isna()), "node_type"
    ] = "grantor"
    nodes.loc[
        (nodes["NTEE1"] != "T") & (nodes["node_type"].isna()), "node_type"
    ] = "grantee"
    nodes.loc[nodes["ein"].isin(both_node_type), "node_type"] = "both"

    # Replace noisey sequence fields with NaN
    nodes.loc[nodes["sequence"].str.len() <= 2, "sequence"] = pd.NA

    # Combine all sequences, drop duplicates, and get rid of repeating words.
    sequence = (
        nodes.fillna("")
        .groupby("ein")["sequence"]
        .apply(set)
        .apply(
            lambda val: " ".join(
                OrderedDict((w, w) for w in " ".join(val).split()).keys()
            )
        )
        .to_dict()
    )
    del nodes["sequence"]

    # Apply sequence mapper to retrieve that information again
    nodes.loc[:, "sequence"] = nodes["ein"].map(sequence)

    # Fill in nodes that do not have a benchmark status (they are not in original study)
    nodes.loc[nodes["benchmark_status"].isna(), "benchmark_status"] = "new"

    # Get unique nodes with relevant fields
    nodes = (
        nodes[NODE_COLS]
        .copy()
        .drop_duplicates(subset=["ein"])
        .sort_values("ein")
        .reset_index(drop=True)
        .fillna("")
    )

    # # Store multiple locations (if present) as a node feature
    # node_locations = (
    #     nodes.drop_duplicates(subset=["ein", "location"])
    #     .groupby("ein")["location"]
    #     .agg(lambda l: "|".join([x for x in l if pd.notna(x)]))
    # )
    # node_locations_mapper = {k: v for k, v in node_locations.to_dict().items() if v}
    # nodes.loc[:, "locations"] = nodes["ein"].map(node_locations_mapper).fillna(pd.NA)

    # # Fill in missing zipcodes with filler 00000 value (to build graph)
    # nodes.loc[nodes["zipcode"].isna(), "zipcode"] = "00000"

    # Pre-processed nodes DF!
    return nodes


@typechecked
def create_edges(
    frac: float = 1.0, seed: int = SEED, verbose: bool = True,
) -> pd.DataFrame:
    """_summary_

    Args:
        frac (float, optional): _description_. Defaults to 1.0.
        seed (int, optional): _description_. Defaults to SEED.
        verbose (bool, optional): _description_. Defaults to True.

    Returns:
        pd.DataFrame: _description_
    """
    # Concatenate all grant descriptions into one
    from collections import OrderedDict

    # Load in grants data to extract edge information
    grants = load_grants(frac=frac, seed=seed, verbose=verbose)
    grants.rename({"grantor_ein": "src", "grantee_ein": "dst"}, axis=1, inplace=True)

    # Drop duplicate edges, and then fill in grant description data
    edges = (
        grants[EDGE_COLS]
        .copy()
        .drop_duplicates(subset=["src", "dst"])
        .reset_index(drop=True)
        .fillna("")
    )

    # # Create edge tuple for networkx ID
    # edges.loc[:, "edge"] = edges.apply(lambda row: (row["src"], row["dst"]), axis=1)

    # # Combine all grant descriptions, drop duplicates, and get rid of repeating words.
    # grant_desc = (
    #     edges.fillna("")
    #     .groupby("edge")["grant_desc"]
    #     .apply(set)
    #     .apply(
    #         lambda val: " ".join(
    #             OrderedDict((w, w) for w in " ".join(val).split()).keys()
    #         )
    #     )
    #     .to_dict()
    # )
    # del edges["grant_desc"]

    # # Get grant description fields again from mapper
    # edges.loc[:, "grant_desc"] = edges["edge"].map(grant_desc)
    # edges.fillna("", inplace=True)
    # del edges["edge"]

    # # Create EIN -> ZIPCODE edge DF
    # zips = (
    #     nodes.copy().drop_duplicates(subset=["ein", "zipcode"]).reset_index(drop=True)
    # )

    # # Apply zip codes & then create
    # zips.loc[:, "edge_type"] = "location"
    # edge_cols = ["source", "destination", "edge_type"]
    # zips.rename({"ein": "source", "zipcode": "destination"}, axis=1, inplace=True)
    # zips = zips[edge_cols].copy()

    # # Combine 2 edge types into 1 DF
    # edges = pd.concat([grants, zips])

    # Drop nodes down to unique EINs, w/ cols: ein, NTEE1, broad_cat, sequence, locations

    # # Add zipcodes as nodes
    # nodes = pd.concat(
    #     [nodes, pd.DataFrame({"ein": nodes["zipcode"].unique().tolist()})]
    # ).reset_index(drop=True)

    # # For zip codes, assign new label type (we'll see how we use it later.) and put in filler, descriptive sequence
    # nodes.loc[nodes["NTEE1"].isna(), "NTEE1"] = "AA"
    # nodes.loc[nodes["broad_cat"].isna(), "broad_cat"] = "XI"
    # nodes.loc[nodes["NTEE1"] == "AA", "sequence"] = "REGION"

    # # Drop duplicates by EIN
    # nodes = (
    #     nodes.drop_duplicates(subset="ein")
    #     .sort_values("ein")
    #     .reset_index(drop=True)
    #     .copy()
    # )

    # nodes = nodes[["ein", "NTEE1", "broad_cat", "locations", "sequence",]].fillna("")
    return edges


@typechecked
def save_graph_dataframes(
    frac: float = 1.0, seed: int = SEED, verbose: bool = True,
) -> None:
    """Save the nodes and edges to parquet.

    Args:
        frac (float, optional): _description_. Defaults to 1.0.
        seed (int, optional): _description_. Defaults to SEED.
        verbose (bool, optional): _description_. Defaults to True.
    """

    nodes = create_nodes(frac=frac, seed=seed, verbose=verbose)
    if verbose:
        print(f"\nCREATING NODES DATAFRAME! w/ cols: {nodes.columns.tolist()}")
        print(f"Shape: {nodes.shape}")

    edges = create_edges(frac=frac, seed=seed, verbose=verbose)
    if verbose:
        print(f"\nCREATING EDGES DATAFRAME! w/ cols: {edges.columns.tolist()}")
        print(f"Shape: {edges.shape}")
    # Save gold DFs!
    save_to_parquet(
        data=nodes[NODE_COLS], cols=NODE_COLS, loc=GRANTS_GOLD_PATH, filename="nodes",
    )
    save_to_parquet(
        data=edges[EDGE_COLS], cols=EDGE_COLS, loc=GRANTS_GOLD_PATH, filename="edges",
    )


@typechecked
def load_nodes(
    frac: float = 1.0, seed: int = SEED, verbose: bool = True,
) -> pd.DataFrame:
    """Load in the nodes data.

    Args:
        frac (float, optional): Return a random fraction of rows from Pandas DataFrame. Defaults to 1.0 (100%).
        seed (int, optional): Random state for reproducibiltiy. Defaults to SEED.
        verbose (bool, optional): If True, print Pandas DataFrame shape. Defaults to True.

    Returns:
        pd.DataFrame: Gold grants DataFrame(s) -> combined or (train, test)
    """
    loc = GRANTS_GOLD_PATH
    filename = "nodes"
    return load_parquet(
        loc=loc, filename=filename, frac=frac, seed=seed, verbose=verbose
    )


@typechecked
def load_edges(
    frac: float = 1.0, seed: int = SEED, verbose: bool = True,
) -> pd.DataFrame:
    """Load in the edges data.

    Args:
        frac (float, optional): Return a random fraction of rows from Pandas DataFrame. Defaults to 1.0 (100%).
        seed (int, optional): Random state for reproducibiltiy. Defaults to SEED.
        verbose (bool, optional): If True, print Pandas DataFrame shape. Defaults to True.

    Returns:
        pd.DataFrame: Gold grants DataFrame(s) -> combined or (train, test)
    """
    loc = GRANTS_GOLD_PATH
    filename = "edges"
    return load_parquet(
        loc=loc, filename=filename, frac=frac, seed=seed, verbose=verbose
    )


@typechecked
def load_graph_dfs(
    frac: float = 1.0, seed: int = SEED, verbose: bool = True
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    return (
        load_nodes(frac=frac, seed=seed, verbose=verbose),
        load_edges(frac=frac, seed=seed, verbose=verbose),
    )


def main():
    """
    Create edges and nodes (gold) of the grants dataset, saving to parquet for easy access.
    """
    save_graph_dataframes(frac=1.0, seed=SEED, verbose=True)


if __name__ == "__main__":
    main()
