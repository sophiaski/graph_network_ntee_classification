# Initialize directory paths, classes, constants, packages, and other methods
from utils import *


from graph_process import load_graph_dfs, load_embs

from torch_geometric.data import Data, HeteroData

import torch_geometric.transforms as T


def stage_graph(
    device: torch.device,
    ntee: bool = False,
    complex_graph: bool = False,
    add_more_targets: bool = False,
    from_saved_model: bool = False,
    hetero_graph: bool = False,
    frac: float = 1.0,
    seed: int = SEED,
    verbose: bool = True,
):

    # Load data
    nodes, edges = load_graph_dfs(
        complex_graph=complex_graph,
        add_more_targets=add_more_targets,
        frac=frac,
        seed=seed,
        verbose=verbose,
    )
    eins, embs = load_embs(
        ntee=ntee,
        complex_graph=complex_graph,
        # add_more_targets=add_more_targets,
        from_saved_model=from_saved_model,
    )

    print(f"Size of eins: {eins.shape}")
    print(f"Size of embeddings (before transformed): {embs.shape}")

    assert len(nodes) == len(eins)
    assert len(nodes) == len(embs)

    #########
    # Nodes #
    #########

    if ntee:
        target = "NTEE1"
    else:
        target = "broad_cat"

    # Generalizing the node types (not sure how to apply new train/test/valid masks for different node types)
    generalizing_node_type = {
        "grantee": "organization",
        "grantor": "organization",
        "both": "organization",
        "region": "region",
    }
    nodes.loc[:, "node_type"] = nodes["node_type"].map(generalizing_node_type)
    del generalizing_node_type
    # ID 2 EIN mapper
    nodes.loc[:, "node_id"] = nodes.index.astype(int)
    ein2id = nodes.set_index("ein")["node_id"].to_dict()

    # Create node feature vector
    missing = nodes[~nodes["ein"].isin(eins)]["ein"].values
    print(f"Are there any EINs missing from the embeddings? {len(missing)}")
    ein2emb = dict(zip(eins, embs))
    id2emb = {ein2id[k]: v for k, v in ein2emb.items()}

    # # Filter the IDS down to the ones not getting removed
    # remove = (
    #     nodes[(nodes["benchmark_status"] == "new") & (nodes["node_type"] != "region")]
    #     .copy()["node_id"]
    #     .values
    # )
    # id2emb = {k: v for k, v in id2emb.items() if k not in remove}

    # nodes = (
    #     nodes[(nodes["benchmark_status"] != "new") | (nodes["node_type"] == "region")]
    #     .copy()
    #     .reset_index(drop=True)
    # )

    # ids = torch.tensor(
    #     [
    #         torch.tensor(x[0]).item()
    #         for x in sorted(id2emb.items(), key=lambda x: int(x[0]))
    #     ],
    #     dtype=torch.long,
    # )
    x = torch.cat(
        [
            torch.tensor(x[1]).unsqueeze(0)
            for x in sorted(id2emb.items(), key=lambda x: int(x[0]))
        ]
    ).to(device)

    from torch.nn.functional import softmax, normalize

    x = normalize(x)

    del missing, ein2emb, id2emb
    # x = torch.rand((len(nodes), embs.shape[1]))
    # x[ids] = embs_parsed
    # print(f"Size of embeddings (after transformed): {x.shape}")

    # Encode groups as numeric target value
    nodes.loc[:, f"{target}_y"] = preprocessing.LabelEncoder().fit_transform(
        nodes[target].values
    )
    # Make unlabeled value == -1
    nodes.loc[nodes[f"{target}_y"] == nodes[f"{target}_y"].max(), f"{target}_y"] = -1

    y = torch.from_numpy(nodes[f"{target}_y"].values)

    #########
    # Edges #
    #########

    # Remove any edges not contained in original dataset

    # For this project, we are not using any additional edge information
    # Create edge weights by dollar amount
    # edges.loc[:, "cash_grant_minmax"] = preprocessing.MinMaxScaler(
    #     feature_range=(0.5, 1)
    # ).fit_transform(edges[["cash_grant_amt"]].fillna("0").astype(int))
    # # Create edge weights by dollar amount
    # edges.loc[:, "tax_period_minmax"] = preprocessing.MinMaxScaler(
    #     feature_range=(0.5, 1)
    # ).fit_transform(edges[["tax_period"]].fillna("2010").astype(int))

    # ID 2 Node Id
    edges.loc[:, "src_id"] = edges["src"].map(ein2id)
    edges.loc[:, "dst_id"] = edges["dst"].map(ein2id)
    # edges = (
    #     edges[(~edges["src_id"].isin(remove)) & (~edges["dst_id"].isin(remove))]
    #     .copy()
    #     .reset_index(drop=True)
    # )

    # # ID 2 EIN mapper
    # nodes.loc[:, "node_id"] = nodes.index.astype(int)
    # new_ein2id = nodes.set_index("ein")["node_id"].to_dict()
    # edges.loc[:, "src_id"] = edges["src"].map(new_ein2id)
    # edges.loc[:, "dst_id"] = edges["dst"].map(new_ein2id)
    # del ein2id, new_ein2id

    # Need to create different node types
    node_type_mapper = nodes["node_type"].to_dict()
    edges.loc[:, "src_node_type"] = edges["src_id"].map(node_type_mapper)
    edges.loc[:, "dst_node_type"] = edges["dst_id"].map(node_type_mapper)
    del node_type_mapper

    edge_index = torch.tensor(
        [edges["src_id"].tolist(), edges["dst_id"].tolist()], dtype=torch.long
    ).to(device)

    # edge_attr = torch.from_numpy(
    #     edges[["cash_grant_minmax", "tax_period_minmax"]].values
    # ).to(device)

    # Spltting out train+val from test
    nodes_trainval = torch.from_numpy(
        nodes[nodes["benchmark_status"] == "train"]["node_id"].values
    )
    nodes_test = torch.from_numpy(
        nodes[nodes["benchmark_status"] == "test"]["node_id"].values
    )
    # target_trainval = torch.from_numpy(
    #     nodes[nodes["benchmark_status"] == "train"][f"{target}_y"].values,
    # )

    nodes_train, nodes_valid = train_test_split(
        nodes_trainval, test_size=0.2, random_state=seed,
    )

    # Create train, validation, and test masks for data
    train_mask = torch.zeros(len(nodes), dtype=torch.bool).to(device)
    valid_mask = torch.zeros(len(nodes), dtype=torch.bool).to(device)
    test_mask = torch.zeros(len(nodes), dtype=torch.bool).to(device)

    train_mask[nodes_train] = True
    valid_mask[nodes_valid] = True
    test_mask[nodes_test] = True

    if hetero_graph:

        data = HeteroData().to(device)

        ##################
        # Add node types #
        ##################

        data["organization"].x = x[
            torch.from_numpy(
                nodes[nodes["node_type"] == "organization"]["node_id"].values
            ).to(device)
        ]  # [num_grantors, num_features_grantors]

        # Train mask
        data["organization"].train_mask = train_mask[
            torch.from_numpy(
                nodes[nodes["node_type"] == "organization"]["node_id"].values
            ).to(device)
        ]

        # Valid mask
        data["organization"].valid_mask = valid_mask[
            torch.from_numpy(
                nodes[nodes["node_type"] == "organization"]["node_id"].values
            ).to(device)
        ]
        # Test mask
        data["organization"].test_mask = test_mask[
            torch.from_numpy(
                nodes[nodes["node_type"] == "organization"]["node_id"].values
            ).to(device)
        ]

        data["region"].x = x[
            torch.from_numpy(
                nodes[nodes["node_type"] == "region"]["node_id"].values
            ).to(device)
        ]  # [num_regions, num_features_regions]

        ####################
        # Add edge indexes #
        ####################

        data["organization", "donates_to", "organization"].edge_index = edge_index[
            :,
            torch.from_numpy(
                edges[
                    (edges["src_node_type"] == "organization")
                    & (edges["dst_node_type"] == "organization")
                ].index.values
            ).to(device),
        ]  # [2, num_edges_bothboth]

        data["organization", "operates_in", "region"].edge_index = edge_index[
            :,
            torch.from_numpy(
                edges[
                    (edges["src_node_type"] == "organization")
                    & (edges["dst_node_type"] == "region")
                ].index.values
            ).to(device),
        ]  # [2, num_edges_operates]

        #####################
        # Add edge features #
        #####################

        # data["organization", "donates_to", "organization"].edge_attr = edge_attr[
        #     :,
        #     torch.from_numpy(
        #         edges[
        #             (edges["src_node_type"] == "organization")
        #             & (edges["dst_node_type"] == "organization")
        #         ].index.values
        #     ).to(device),
        # ]  # [num_edges_donates, num_features_donates]

        # data["organization", "operates_in", "region"].edge_attr = edge_attr[
        #     :,
        #     torch.from_numpy(
        #         edges[
        #             (edges["src_node_type"] == "organization")
        #             & (edges["dst_node_type"] == "region")
        #         ].index.values
        #     ).to(device),
        # ]  # [num_edges_org_operates, num_features_operates]

        data = T.ToUndirected()(data)
        data = T.AddSelfLoops()(data)
        # data = T.NormalizeFeatures()(data)

    else:

        data = Data().to(device)
        data.x = x
        # data.edge_attr = edge_attr
        data.edge_index = edge_index
        data.y = y
        data.train_mask = train_mask
        data.test_mask = test_mask
        data.valid_mask = valid_mask
        # num_features = x.shape[1]
        # num_classes = y.max().item()

        data = T.ToUndirected()(data)
        data = T.AddSelfLoops()(data)
        edge_attr = torch.ones(len(data.edge_index[0]), 2) * 0.5
        edge_attr[: data.edge_attr.shape[0], : data.edge_attr.shape[1]] = data.edge_attr
        data["edge_attr"] = edge_attr
        # edge_attr_data = torch.from_numpy(
        #     edges[["cash_grant_minmax", "tax_period_minmax"]].values
        # )

        # edge_attr[: edge_attr_data.shape[0], : edge_attr_data.shape[1]] = edge_attr_data
        # data = T.NormalizeFeatures()(data)

    return data


# from transformers import BertModel, BertTokenizer

# def create_attribs(df, key_col, val_col):
#     return pd.Series(df[val_col].values, index=df[key_col]).dropna().to_dict()

# @typechecked
# def build_PyG_graph() -> torch_geomtric.Graph:
#     nodes, edges = load_graph_dataframes()
#     model = BertModel.from_pretrained("bert-base-uncased")
#     tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

#     # Convert target labels to integers, save as "y"

#     # Convert sequence to BERT embeddings, save as "x"
#     for index, row in nodes.iterrows():
#         seq = row["sequence"]
#         input = tokenizer(
#             seq,
#             add_special_tokens=True,
#             truncation="longest_first",
#             padding="max_length",
#             max_length=128,
#             return_attention_mask=True,
#             return_tensors="pt",
#         )

#         # nodes[index, "x"] =

#     return

# @typechecked
# def create_nx_graph() -> nx.Graph:

#     nodes, edges = load_graph_dataframes()
#     # Create edge tuple for networkx ID
#     G = nx.from_pandas_edgelist(edges, source="source", target="destination")

#     for col in ["cash_grant_amt", "edge_type", "grant_desc", "tax_period"]:
#         nx.set_edge_attributes(G, create_attribs(edges, "edge", col), col)

#     for col in ["NTEE1", "broad_cat", "locations", "sequence"]:
#         nx.set_node_attributes(G, create_attribs(nodes, "ein", col), col)

#     return G


# @typechecked
# def save_nx_graph_to_json(G: nx.Graph) -> None:
#     save_to_json(data=G, loc=GRAPH_GOLD_PATH, filename="test")


def main():
    return None


if __name__ == "__main__":
    main()

    # FOR EXPLORING OTHER NODE TYPES LATER

    # # Add node types
    # data["grantor"].x = x[
    #     torch.from_numpy(nodes[nodes["node_type"] == "grantor"]["node_id"].values)
    # ]  # [num_grantors, num_features_grantors]
    # data["grantee"].x = x[
    #     torch.from_numpy(nodes[nodes["node_type"] == "grantee"]["node_id"].values)
    # ]  # [num_grantees, num_features_grantees]
    # data["both"].x = x[
    #     torch.from_numpy(nodes[nodes["node_type"] == "both"]["node_id"].values)
    # ]  # [num_both, num_features_both]
    # data["region"].x = x[
    #     torch.from_numpy(nodes[nodes["node_type"] == "region"]["node_id"].values)
    # ]  # [num_regions, num_features_regions]

    # # Add edge indexs
    # data["grantor", "donates_to", "grantee"].edge_index = edge_index[
    #     :,
    #     edges[
    #         (edges["src_node_type"] == "grantor")
    #         & (edges["dst_node_type"] == "grantee")
    #     ].index.values,
    # ]  # [2, num_edges_grantorgrantee]
    # data["both", "donates_to", "grantee"].edge_index = edge_index[
    #     :,
    #     edges[
    #         (edges["src_node_type"] == "both")
    #         & (edges["dst_node_type"] == "grantee")
    #     ].index.values,
    # ]  # [2, num_edges_bothgrantee]
    # data["both", "donates_to", "both"].edge_index = edge_index[
    #     :,
    #     edges[
    #         (edges["src_node_type"] == "both") & (edges["dst_node_type"] == "both")
    #     ].index.values,
    # ]  # [2, num_edges_bothboth]
    # data["grantor", "operates_in", "region"].edge_index = edge_index[
    #     :,
    #     edges[
    #         (edges["src_node_type"] == "grantor")
    #         & (edges["dst_node_type"] == "region")
    #     ].index.values,
    # ]  # [2, num_edges_bothgrantee]
    # data["grantee", "operates_in", "region"].edge_index = edge_index[
    #     :,
    #     edges[
    #         (edges["src_node_type"] == "grantee")
    #         & (edges["dst_node_type"] == "region")
    #     ].index.values,
    # ]  # [2, num_edges_bothgrantee]
    # data["both", "operates_in", "region"].edge_index = edge_index[
    #     :,
    #     edges[
    #         (edges["src_node_type"] == "both")
    #         & (edges["dst_node_type"] == "region")
    #     ].index.values,
    # ]  # [2, num_edges_bothgrantee]

    # # Add edge features
    # data["grantor", "donates_to", "grantee"].edge_attr = edge_attr[
    #     :,
    #     edges[
    #         (edges["src_node_type"] == "grantor")
    #         & (edges["dst_node_type"] == "grantee")
    #     ].index.values,
    # ]  # [num_edges_grantorgrantee, num_features_grantorgrantee]
    # data["both", "donates_to", "grantee"].edge_attr = edge_attr[
    #     :,
    #     edges[
    #         (edges["src_node_type"] == "both")
    #         & (edges["dst_node_type"] == "grantee")
    #     ].index.values,
    # ]  # [num_edges_bothgrantee, num_features_bothgrantee]
    # data["both", "donates_to", "both"].edge_attr = edge_attr[
    #     :,
    #     edges[
    #         (edges["src_node_type"] == "both") & (edges["dst_node_type"] == "both")
    #     ].index.values,
    # ]  # [num_edges_bothgrantee, num_features_bothgrantee]
    # data["grantee", "operates_in", "region"].edge_attr = edge_attr[
    #     :,
    #     edges[
    #         (edges["src_node_type"] == "grantee")
    #         & (edges["dst_node_type"] == "region")
    #     ].index.values,
    # ]  # [num_edges_bothgrantee, num_features_bothgrantee]
    # data["grantor", "operates_in", "region"].edge_attr = edge_attr[
    #     :,
    #     edges[
    #         (edges["src_node_type"] == "grantor")
    #         & (edges["dst_node_type"] == "region")
    #     ].index.values,
    # ]  # [num_edges_bothgrantee, num_features_bothgrantee]
    # data["both", "operates_in", "region"].edge_attr = edge_attr[
    #     :,
    #     edges[
    #         (edges["src_node_type"] == "both")
    #         & (edges["dst_node_type"] == "region")
    #     ].index.values,
    # ]  # [num_edges_bothgrantee, num_features_bothgrantee]

