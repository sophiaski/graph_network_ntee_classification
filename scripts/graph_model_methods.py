"""All of the methods used to explore GNNs for text classification on a heterogeneous graph"""

# Initialize directory paths, classes, constants, packages, and other methods
from utils import *

from graph_process import load_graph_dfs

# Preparing data
from sklearn import preprocessing


def stage_simple_graph():

    # Load data
    nodes, edges = load_graph_dfs()

    #########
    # Nodes #
    #########

    # ID 2 EIN mapper
    nodes.loc[:, "node_id"] = nodes.index.astype(int)
    ein2id = nodes.set_index("ein")["node_id"].to_dict()
    id2ein = {v: k for k, v in ein2id.items()}

    # Node is mapped to index of processed dataframe. Store.
    save_to_json(data=id2ein, loc=EMBEDDINGS_PATH, filename="simple_id2ein")

    # Convert target labels to integers, save as "y"
    # Encode groups as numeric target value
    nodes.loc[:, "broad_cat_y"] = preprocessing.LabelEncoder().fit_transform(
        nodes["broad_cat"].values
    )
    nodes.loc[:, "NTEE1_y"] = preprocessing.LabelEncoder().fit_transform(
        nodes["NTEE1"].values
    )

    #########
    # Edges #
    #########

    # Create edge weights by dollar amount
    edges.loc[:, "cash_grant_minmax"] = preprocessing.MinMaxScaler(
        feature_range=(0.5, 1)
    ).fit_transform(edges[["cash_grant_amt"]].fillna("0").astype(int))
    # Create edge weights by dollar amount
    edges.loc[:, "tax_period_minmax"] = preprocessing.MinMaxScaler(
        feature_range=(0.5, 1)
    ).fit_transform(edges[["tax_period"]].fillna("2010").astype(int))

    # ID 2 Node Id
    edges.loc[:, "src_id"] = edges["src"].map(ein2id)
    edges.loc[:, "dst_id"] = edges["dst"].map(ein2id)

    ## TARGET
    broadtarget2group = nodes.groupby("broad_cat_y")["broad_cat"].first().to_dict()
    nteetarget2group = nodes.groupby("NTEE1_y")["NTEE1"].first().to_dict()
    # Encoded ids to group label mappers
    save_to_json(data=broadtarget2group, loc=EMBEDDINGS_PATH, filename="simple_broad")
    save_to_json(data=nteetarget2group, loc=EMBEDDINGS_PATH, filename="simple_ntee1")

    return nodes, edges


# Set up for training on optimal BERT
# Then without

# GNNs
# GCN
# GAT

# Make graph more complicated if scores still stuck
# User hyperparameters from Coras dataset
