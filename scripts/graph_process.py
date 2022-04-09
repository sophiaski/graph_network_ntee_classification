# Initialize directory paths, classes, constants, packages, and other methods
from matplotlib.cbook import flatten
from utils import *

# Get other methods to load in cleaned data
from grants_process import load_grants
from benchmark_process import load_benchmark

# Get mapper that converts from NTEE1 to Broad category
from collections import ChainMap, OrderedDict


@typechecked
def create_nodes(
    complex_graph: bool = False,
    flatten_graph: bool = False,
    frac: float = 1.0,
    seed: int = SEED,
    verbose: bool = True,
) -> pd.DataFrame:
    """Create the nodes DataFrame.

    Args:
        complex_graph (bool, optional): Strategy for building a complex graph with location nodes. Defaults to False.
        flatten_graph (bool, optional): Strategy for including some of the surrounding node features directly in sequence field. Defaults to False.
        frac (float, optional): Return a random fraction of rows from Pandas DataFrame. Defaults to 1.0 (100%).
        seed (int, optional): Random state for reproducibiltiy. Defaults to SEED.
        verbose (bool, optional): Print Pandas DataFrame shape. Defaults to True.

    Returns:
        pd.DataFrame: Node DataFrame.
    """

    ################
    # COMBINE DATA #
    ################

    # Load in two datasets
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

    ################
    # MISSING DATA #
    ################

    # If organization name is missing, fill in values w/ taxpayer_name
    nodes.loc[nodes["org"].isna() & nodes["taxpayer_name"].notna(), "org"] = nodes[
        nodes["org"].isna() & nodes["taxpayer_name"].notna()
    ]["taxpayer_name"]

    # Similarly, if sequence information is missing, fill in values w/ the organization name
    nodes.loc[nodes["sequence"].isna() & nodes["org"].notna(), "sequence"] = nodes[
        nodes["sequence"].isna() & nodes["org"].notna()
    ]["org"]

    #################################
    # DO WE WANT TO REALLY DO THIS? #
    #################################

    # # See if we can fill in more NTEE1 labels from BMF data
    # mapper = load_json(loc=f"{BRONZE_PATH}cleanup/", filename=f"ein_NTEE1")
    # nodes.loc[nodes["NTEE1"].isna(), "NTEE1"] = (
    #     nodes[nodes["NTEE1"].isna()]["ein"].map(mapper).fillna("Z")
    # )
    # nodes.loc[:, "NTEE1"] = nodes[
    #     "NTEE1"
    # ].str.upper()  # Some of the values are lowercase?

    # # Then, map the NTEE1 codes to the broad categories
    # BROAD_CAT_MAPPER = dict(
    #     ChainMap(*[{letter: k for letter in v} for k, v in BROAD_CAT_DICT.items()])
    # )
    # nodes.loc[nodes["broad_cat"].isna(), "broad_cat"] = nodes[
    #     nodes["broad_cat"].isna()
    # ]["NTEE1"].map(BROAD_CAT_MAPPER)

    ############
    # OR THIS? #
    ############

    # # Leverage grant description data to fill in missing sequences.
    # grant_desc_mapper = (
    #     grants.loc[
    #         grants["grantee_ein"].isin(
    #             nodes[nodes["sequence"].isna()]["ein"].tolist()
    #         ),
    #         ["grantee_ein", "grant_desc"],
    #     ]
    #     .sort_values("grant_desc")
    #     .drop_duplicates(subset="grantee_ein")
    #     .set_index("grantee_ein")["grant_desc"]
    #     .to_dict()
    # )
    # nodes.loc[nodes["sequence"].isna(), "sequence"] = nodes[
    #     nodes["sequence"].isna()
    # ]["ein"].map(grant_desc_mapper)

    #####################
    # NODE_TYPE FEATURE #
    #####################

    # Create node_type feature (grantor, grantee, both)
    # missing_node_type = set(nodes[nodes["node_type"].isna()]["ein"].unique())
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

    ###########################################
    # ADD  GRAPH INFORMATION TO SEQUENCE DATA #
    ###########################################

    if flatten_graph:

        ## NODE_TYPE

        # Append node+type information to beginning of sequence so its detected in classifier
        nodes.loc[:, "sequence"] = (
            nodes["node_type"].map(
                {
                    "grantor": "Funder.".upper(),
                    "grantee": "Grant Recipient.".upper(),
                    "both": "Both funder and recipient.".upper(),
                }
            )
            + " "
            + nodes["sequence"]
        )

    ## GRANT DESCRIPTIONS

    if flatten_graph or complex_graph:

        ## Include grant descriptions in sequence data.
        grants.fillna("", inplace=True)

        # Create unique identiier for cleaning up grant descriptions
        grants.loc[:, "edge"] = grants["grantee_ein"] + grants["grantor_ein"]

        # Combine all grant descriptions, drop duplicates, and get rid of repeating words.

        grant_desc = (
            grants.groupby(by="edge")["grant_desc"]
            .apply(set)
            .apply(
                lambda val: " ".join(
                    OrderedDict((w, w) for w in " ".join(val).split()).keys()
                )
            )
            .to_dict()
        )
        del grants["grant_desc"]

        # Get cleaned grant description fields from mapper
        grants.loc[:, "grant_desc"] = grants["edge"].map(grant_desc).fillna(pd.NA)

        ## ADD TO SEQUENCE

        # Collect all unique grant descriptions for each grantee's EIN and then append to the end of the sequence in human readable format.
        # grants.loc[:, "grant_desc"] = grants["grant_desc"]
        grant_desc_mapper = (
            grants.drop_duplicates(subset=["grantee_ein", "grant_desc"])
            .groupby(by="grantee_ein")["grant_desc"]
            .agg(lambda l: "|".join([x for x in l if pd.notna(x)]))
            .to_dict()
        )

        grant_desc_mapper_str = {
            k: f"They received funding for their programs on {readable_list(v.split('|'))}.".upper()
            for k, v in grant_desc_mapper.items()
            if v
        }
        # Append locations to end of sequence so its detected in classifier
        # nodes.loc[:, "sequence"] = (
        #     nodes["sequence"] + " " + nodes["ein"].map(grant_desc_mapper_str)
        # )

        nodes.loc[:, "sequence"] = nodes.apply(
            lambda row: f"{row['sequence']}. {grant_desc_mapper_str[row['ein']]}"
            if grant_desc_mapper_str.get(row["ein"])
            else row["sequence"],
            axis=1,
        )

    if flatten_graph:

        ## LOCATIONS

        # Store multiple locations (if present) in sequence field
        node_locations = (
            nodes.copy()
            .drop_duplicates(subset=["ein", "location"])
            .groupby("ein")["location"]
            .agg(lambda l: "|".join([x for x in l if pd.notna(x)]))
        )
        node_locations_mapper = {
            k: f"Operates out of {readable_list(v.split('|'))}.".upper()
            for k, v in node_locations.to_dict().items()
            if v
        }
        # Append locations to end of sequence so its detected in classifier
        nodes.loc[:, "sequence"] = nodes.apply(
            lambda row: f"{row['sequence']}. {node_locations_mapper[row['ein']]}"
            if node_locations_mapper.get(row["ein"])
            else row["sequence"],
            axis=1,
        )

    ###########################
    # ADD COMPLEXITY TO NODES #
    ###########################

    if complex_graph:

        ## CREATE REGION NODES

        # Fill in missing zipcodes with filler 00000 value
        nodes.loc[nodes["zipcode"].isna(), "zipcode"] = "00000"

        # Create zipcode dataframe with string format: "Operating zipcode: 95816"
        zipcodes = (
            nodes[["zipcode"]]
            .copy()
            .rename({"zipcode": "ein"}, axis=1)
            .drop_duplicates(subset="ein")
        )
        zipcodes.loc[:, "sequence"] = "Operating zipcode: " + zipcodes["ein"]
        zipcodes.loc[:, "node_type"] = "region"

        nodes = pd.concat([nodes, zipcodes])

        ## PROGRAM NODES

        # Not for this project!

    ##################
    # CLEAN SEQUENCE #
    ##################

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

    ##########################################
    # UNLABELED NODES - NOT USED IN TRAINING #
    ##########################################

    # Fill in nodes that do not have a benchmark status (they are not in original study)
    nodes.loc[nodes["benchmark_status"].isna(), "benchmark_status"] = "new"
    nodes.loc[nodes["NTEE1"].isna(), "NTEE1"] = "Z"
    nodes.loc[nodes["broad_cat"].isna(), "broad_cat"] = "X"

    ##########
    # OUTPUT #
    ##########

    # Output nodes sorted alphabetically, descending by benchmark_status (train, test, new) and then alphabetically, ascending by ein
    return (
        nodes[NODE_COLS]
        .copy()
        .drop_duplicates(subset=["ein"])
        .sort_values(by=["benchmark_status", "ein"], ascending=[False, True])
        .reset_index(drop=True)
        .fillna("")
    )


@typechecked
def create_edges(
    complex_graph: bool = False,
    frac: float = 1.0,
    seed: int = SEED,
    verbose: bool = True,
) -> pd.DataFrame:
    """_summary_

    Args:
        frac (float, optional): _description_. Defaults to 1.0.
        seed (int, optional): _description_. Defaults to SEED.
        verbose (bool, optional): _description_. Defaults to True.

    Returns:
        pd.DataFrame: _description_
    """
    # Load in grants data to extract edge information
    grants = load_grants(frac=frac, seed=seed, verbose=verbose)

    # Make copy of grants as final DF
    edges = grants.copy().rename({"grantor_ein": "src", "grantee_ein": "dst"}, axis=1)

    # Donation edge_type
    edges.loc[:, "edge_type"] = "donates_to"

    if complex_graph:

        # This will generate two types of edges: donates_to & operates_in
        # Idea here is to make the graph more fully connected.

        ## REGION EDGES
        # Fill in missing zipcodes with filler 00000 value
        for typ in ["grantor", "grantee"]:
            edges.loc[edges[f"{typ}_zipcode"].isna(), f"{typ}_zipcode"] = "00000"

        # Create region edge dataframes, each organization pointing to their region node
        zipcodes_src = (
            edges[["src", "grantor_zipcode"]]
            .copy()
            .rename({"grantor_zipcode": "dst"}, axis=1)
            .drop_duplicates(subset=["src", "dst"])
        )
        zipcodes_src.loc[:, "edge_type"] = "operates_in"
        zipcodes_dst = (
            edges[["dst", "grantee_zipcode"]]
            .copy()
            .rename({"dst": "src"}, axis=1)
            .rename({"grantee_zipcode": "dst"}, axis=1)
            .drop_duplicates(subset=["src", "dst"])
        )
        zipcodes_dst.loc[:, "edge_type"] = "operates_in"

        ## Combine
        edges = pd.concat([edges, zipcodes_src, zipcodes_dst])

    ##########
    # OUTPUT #
    ##########

    # Output edges sorted alphabetically, ascending by edge_type (donates_to, operates_in) and then alphabetically, ascending by src ein.
    return (
        edges[EDGE_COLS]
        .copy()
        .drop_duplicates(subset=["src", "dst"])
        .sort_values(by=["edge_type", "src"], ascending=[True, True])
        .reset_index(drop=True)
        .fillna("")
    )


@typechecked
def save_graph_dataframes(
    complex_graph: bool = False,
    flatten_graph: bool = False,
    frac: float = 1.0,
    seed: int = SEED,
    verbose: bool = True,
) -> None:
    """Save the nodes and edges to parquet.

    Args:
        frac (float, optional): _description_. Defaults to 1.0.
        seed (int, optional): _description_. Defaults to SEED.
        verbose (bool, optional): _description_. Defaults to True.
    """

    nodes = create_nodes(
        complex_graph=complex_graph,
        flatten_graph=flatten_graph,
        frac=frac,
        seed=seed,
        verbose=verbose,
    )
    if verbose:
        print(f"\nCREATING NODES DATAFRAME! w/ cols: {nodes.columns.tolist()}")
        print(f"Shape: {nodes.shape}")

    edges = create_edges(
        complex_graph=complex_graph, frac=frac, seed=seed, verbose=verbose
    )
    if verbose:
        print(f"\nCREATING EDGES DATAFRAME! w/ cols: {edges.columns.tolist()}")
        print(f"Shape: {edges.shape}")

    # Save dataframes
    if complex_graph:
        filename = "complex"
    else:
        filename = "simple"

    save_to_parquet(
        data=edges[EDGE_COLS],
        cols=EDGE_COLS,
        loc=GRAPH_GOLD_PATH,
        filename=f"edges_{filename}",
    )
    if flatten_graph:
        filename += "_flattened"

    save_to_parquet(
        data=nodes[NODE_COLS],
        cols=NODE_COLS,
        loc=GRAPH_GOLD_PATH,
        filename=f"nodes_{filename}",
    )


@typechecked
def load_nodes(
    complex_graph: bool = False,
    flatten_graph: bool = False,
    frac: float = 1.0,
    seed: int = SEED,
    verbose: bool = True,
) -> pd.DataFrame:
    """Load in the nodes data.

    Args:
        frac (float, optional): Return a random fraction of rows from Pandas DataFrame. Defaults to 1.0 (100%).
        seed (int, optional): Random state for reproducibiltiy. Defaults to SEED.
        verbose (bool, optional): If True, print Pandas DataFrame shape. Defaults to True.

    Returns:
        pd.DataFrame: Gold grants DataFrame(s) -> combined or (train, test)
    """
    filename = "nodes"
    if complex_graph:
        filename += "_complex"
    else:
        filename += "_simple"
    if flatten_graph:
        filename += "_flattened"

    return load_parquet(
        loc=GRAPH_GOLD_PATH, filename=filename, frac=frac, seed=seed, verbose=verbose
    )


@typechecked
def load_edges(
    complex_graph: bool = False,
    frac: float = 1.0,
    seed: int = SEED,
    verbose: bool = True,
) -> pd.DataFrame:
    """Load in the edges data.

    Args:
        frac (float, optional): Return a random fraction of rows from Pandas DataFrame. Defaults to 1.0 (100%).
        seed (int, optional): Random state for reproducibiltiy. Defaults to SEED.
        verbose (bool, optional): If True, print Pandas DataFrame shape. Defaults to True.

    Returns:
        pd.DataFrame: Gold grants DataFrame(s) -> combined or (train, test)
    """
    filename = "edges"
    if complex_graph:
        filename += "_complex"
    else:
        filename += "_simple"
    return load_parquet(
        loc=GRAPH_GOLD_PATH, filename=filename, frac=frac, seed=seed, verbose=verbose
    )


@typechecked
def load_graph_dfs(
    complex_graph: bool = False,
    flatten_graph: bool = False,
    frac: float = 1.0,
    seed: int = SEED,
    verbose: bool = True,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    return (
        load_nodes(
            complex_graph=complex_graph,
            flatten_graph=flatten_graph,
            frac=frac,
            seed=seed,
            verbose=verbose,
        ),
        load_edges(complex_graph=complex_graph, frac=frac, seed=seed, verbose=verbose,),
    )


def main():
    """
    Create edges and nodes (gold) of the grants dataset, saving to parquet for easy access.
    """
    # Simple graph, no flattening of graph-based features (AKA the benchmark)
    save_graph_dataframes(
        complex_graph=False, flatten_graph=False, frac=1.0, seed=SEED, verbose=True,
    )
    # Simple graph, w/ flattening of graph-based features
    save_graph_dataframes(
        complex_graph=False, flatten_graph=True, frac=1.0, seed=SEED, verbose=True,
    )
    # Complex graph, with location nodes and edges
    save_graph_dataframes(
        complex_graph=True, flatten_graph=False, frac=1.0, seed=SEED, verbose=True,
    )


if __name__ == "__main__":
    main()
