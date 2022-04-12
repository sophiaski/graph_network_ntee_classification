# Initialize directory paths, classes, constants, packages, and other methods
from os import remove
from utils import *


from graph_process import load_graph_dfs, load_embs

from torch_geometric.data import Data, HeteroData

import torch_geometric.transforms as T

from torch.nn.functional import normalize


def create_bow_hetero_graph(
    device: torch.device,
    ntee: bool = False,
    complex_graph: bool = False,
    add_more_targets: bool = False,
    remove_excessive_connections: bool = False,
    verbose: bool = True,
) -> HeteroData:
    # Load data
    nodes, edges = load_graph_dfs(
        complex_graph=complex_graph,
        add_more_targets=add_more_targets,
        frac=1.0,
        seed=SEED,
        verbose=verbose,
    )

    #########
    # NODES #
    #########

    x = load_embs(
        ntee=False, complex_graph=complex_graph, from_saved_model=False, use_bow=True,
    ).float()
    if verbose:
        print(f"Size of embeddings: {x.shape}")

    # Specifiy classification target
    if ntee:
        target = "NTEE1"
    else:
        target = "broad_cat"
    if verbose:
        print(f"Encoding the '{target}' label for this classification task.")

    # Create node types
    generalizing_node_type = {
        "grantee": "organization",
        "grantor": "organization",
        "both": "organization",
        "region": "region",
    }
    nodes.loc[:, "node_type"] = nodes["node_type"].map(generalizing_node_type)

    if verbose:
        print(f"Creating {len(set(generalizing_node_type.values()))} node types.")
    del generalizing_node_type

    # Do we want a slimmed down network only between organizations included in benchmark?
    if remove_excessive_connections:
        # Assign node ids to embedding vectors
        # id2emb = {idx: emb_vector for idx, emb_vector in enumerate(x)}
        # See where in the database
        # Filter the IDS down to the ones not getting remove
        node_ids = (
            nodes[
                (nodes["benchmark_status"] != "new") | (nodes["node_type"] == "region")
            ]
            .copy()
            .index.values
        )
        if verbose:
            print(
                f"Removing {len(nodes)-len(node_ids):,} nodes & embeddings not found in the original benchmark paper."
            )
        # Update nodes with correct IDs and reset index
        nodes = nodes[nodes.index.isin(node_ids)].copy().reset_index(drop=True)

        # Update embs
        x = x[node_ids]

        # Update this key value store for embeddings, filter out the nodes set for removal
        # id2emb = {k: v for k, v in id2emb.items() if k not in new_non_region_nodes}

        # Update embeddings vector, replacing with sorted key-value store by node_id
        # x = torch.cat(
        #     [
        #         torch.tensor(x[1]).unsqueeze(0)
        #         for x in sorted(id2emb.items(), key=lambda x: int(x[0]))
        #     ]
        # )
        if verbose:
            print(
                f"After removing excessive nodes, we are left with embeddings of size {x.shape} and {len(nodes)} nodes."
            )
        # del new_non_region_nodes, id2emb

    # Normalize the bow one-hot embeddings
    x = normalize(x)

    # Encode groups as numeric target value
    nodes.loc[:, f"{target}_y"] = preprocessing.LabelEncoder().fit_transform(
        nodes[target].values
    )
    # # Make unlabeled value == -1
    # nodes.loc[nodes[f"{target}_y"] == nodes[f"{target}_y"].max(), f"{target}_y"] = -1

    y = torch.from_numpy(nodes[f"{target}_y"].values)

    ########################
    # TRAIN/VAL/TEST MASKS #
    ########################
    # ID 2 EIN mapper
    nodes.loc[:, "node_id"] = nodes.index.astype(int)
    ein2id = nodes.set_index("ein")["node_id"].to_dict()

    # Spltting out train+val from test
    nodes_trainval = torch.from_numpy(
        nodes[nodes["benchmark_status"] == "train"]["node_id"].values
    )
    nodes_test = torch.from_numpy(
        nodes[nodes["benchmark_status"] == "test"]["node_id"].values
    )
    nodes_train, nodes_val = train_test_split(
        nodes_trainval, test_size=0.2, random_state=SEED,
    )
    if verbose:
        print(
            f"Split intro train/val/test: ({len(nodes_train)}, {len(nodes_val)}, {len(nodes_test)})"
        )
    # Create train, validation, and test masks for data
    train_mask = torch.zeros(len(nodes), dtype=torch.bool)
    val_mask = torch.zeros(len(nodes), dtype=torch.bool)
    test_mask = torch.zeros(len(nodes), dtype=torch.bool)
    train_mask[nodes_train] = True
    val_mask[nodes_val] = True
    test_mask[nodes_test] = True

    #########
    # EDGES #
    #########

    # ID 2 Node Id
    edges.loc[:, "src_id"] = edges["src"].map(ein2id)
    edges.loc[:, "dst_id"] = edges["dst"].map(ein2id)

    if remove_excessive_connections:
        edges = (
            edges.copy()
            .dropna(axis=0, how="any", subset=["src_id", "dst_id"])
            .reset_index(drop=True)
        )

    # Need to create different node types
    node_type_mapper = nodes["node_type"].to_dict()
    edges.loc[:, "src_node_type"] = edges["src_id"].map(node_type_mapper)
    edges.loc[:, "dst_node_type"] = edges["dst_id"].map(node_type_mapper)
    del node_type_mapper

    edge_index = torch.tensor(
        [edges["src_id"].tolist(), edges["dst_id"].tolist()], dtype=torch.long
    )

    data = HeteroData()

    ##################
    # Add node types #
    ##################

    data["organization"].x = x[
        torch.from_numpy(nodes[nodes["node_type"] == "organization"].index.values)
    ]  # [num_grantors, num_features_grantors]

    data["organization"].y = y[
        torch.from_numpy(nodes[nodes["node_type"] == "organization"].index.values)
    ]  # [num_grantors, num_features_grantors]

    # Train mask
    data["organization"].train_mask = train_mask[
        torch.from_numpy(nodes[nodes["node_type"] == "organization"].index.values)
    ]

    # Valid mask
    data["organization"].val_mask = val_mask[
        torch.from_numpy(nodes[nodes["node_type"] == "organization"].index.values)
    ]
    # Test mask
    data["organization"].test_mask = test_mask[
        torch.from_numpy(nodes[nodes["node_type"] == "organization"].index.values)
    ]

    data["region"].x = x[
        torch.from_numpy(nodes[nodes["node_type"] == "region"].index.values)
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
        ),
    ]  # [2, num_edges_bothboth]

    data["organization", "operates_in", "region"].edge_index = edge_index[
        :,
        torch.from_numpy(
            edges[
                (edges["src_node_type"] == "organization")
                & (edges["dst_node_type"] == "region")
            ].index.values
        ),
    ]  # [2, num_edges_operates]

    # data["organization", "donates_to", "organization"].edge_attr = torch.ones(
    #     2, len(edges)
    # )  # [2, num_edges_bothboth]

    # if complex_graph:
    #     data["organization", "operates_in", "region"].edge_attr = (
    #         torch.ones(2, len(edges)) * 0.5
    #     )  # [2, num_edges_operates]

    del nodes, edges

    # data = T.NormalizeFeatures()(data)
    data = T.AddSelfLoops()(data)
    data = T.ToUndirected(merge=False)(data)

    return data


def stage_graph(
    device: torch.device,
    ntee: bool = False,
    complex_graph: bool = False,
    add_more_targets: bool = False,
    from_saved_model: bool = False,
    remove_excessive_connections: bool = False,
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
    print(f"Size of embeddings: {embs.shape}")

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

    del eins, embs

    if remove_excessive_connections:
        # Filter the IDS down to the ones not getting remove
        remove = (
            nodes[
                (nodes["benchmark_status"] == "new") & (nodes["node_type"] != "region")
            ]
            .copy()
            .index.values
        )
        id2emb = {k: v for k, v in id2emb.items() if k not in remove}
        nodes = nodes[~nodes.index.isin(remove)].copy()
        print("Nodes df after excessive removal: ", nodes.shape)
        # nodes = (
        #     nodes[
        #         (nodes["benchmark_status"] != "new") | (nodes["node_type"] == "region")
        #     ]
        #     .copy()
        #     .reset_index(drop=True)
        # )
        regions = nodes[nodes["node_type"] == "region"].copy()
        regions_lst = regions.index.values
        region_emb = id2emb[50]  # torch tensor

        id2emb = {k: v for k, v in id2emb.items() if k not in regions_lst}
        nodes = nodes[~nodes.index.isin(regions_lst)].copy()
        print("Nodes df after region removal: ", nodes.shape)
        # Now we drop zip codes!
        # Updating zipcode to state
        zip_2_state = pd.read_json("../../US-Zip-Codes-JSON/USCities.json")[
            ["zip_code", "state"]
        ]
        zip_2_state["zip_code_len"] = zip_2_state["zip_code"].astype(str).str.len()
        zip_2_state["zip"] = zip_2_state["zip_code"].astype(str)
        zip_2_state.loc[zip_2_state["zip_code_len"] == 3, "zip"] = (
            "00" + zip_2_state["zip"]
        )
        zip_2_state.loc[zip_2_state["zip_code_len"] == 4, "zip"] = (
            "0" + zip_2_state["zip"]
        )
        zip2state = zip_2_state[["zip", "state"]].set_index("zip")["state"].to_dict()
        zip2state["00000"] = "??"
        print(len(zip2state))
        regions.loc[regions["node_type"] == "region", "ein"] = regions["ein"].map(
            zip2state
        )
        regions.drop_duplicates(subset="ein", inplace=True)
        nodes = pd.concat([nodes, regions]).reset_index(drop=True)
        print(regions["ein"].tolist())
        region_embs = torch.cat([region_emb.unsqueeze(0)] * len(regions), dim=0)
        print("REGION EMBS:", region_embs.shape)
        print(f"Nodes DF: {nodes.shape}")
        # ids = torch.tensor(
        #     [
        #         torch.tensor(x[0]).item()
        #         for x in sorted(id2emb.items(), key=lambda x: int(x[0]))
        #     ],
        #     dtype=torch.long,
        # )
        nodes_embs = torch.cat(
            [
                torch.tensor(x[1]).unsqueeze(0)
                for x in sorted(id2emb.items(), key=lambda x: int(x[0]))
            ]
        )
        print("node embs shape", nodes_embs.shape)
        x = normalize(torch.cat((nodes_embs, region_embs), dim=0))
        print(x.shape)

    del missing, ein2emb, id2emb, nodes_embs, region_embs, regions, zip_2_state
    # x = torch.rand((len(nodes), embs.shape[1]))
    # x[ids] = embs_parsed
    # print(f"Size of embeddings (after transformed): {x.shape}")

    # Encode groups as numeric target value
    nodes.loc[:, f"{target}_y"] = preprocessing.LabelEncoder().fit_transform(
        nodes[target].values
    )
    # Make unlabeled value == -1
    # nodes.loc[nodes[f"{target}_y"] == nodes[f"{target}_y"].max(), f"{target}_y"] = -1
    y = torch.from_numpy(nodes[f"{target}_y"].values)

    #########
    # Edges #
    #########

    # ID 2 EIN mapper
    nodes.loc[:, "node_id"] = nodes.index.astype(int)
    ein2id = nodes.set_index("ein")["node_id"].to_dict()

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

    # Add region nodes
    # Convert edge zips to states as well
    edges.loc[edges["src"].str.len() == 5, "src"] = edges["src"].map(zip2state)
    edges.loc[edges["dst"].str.len() == 5, "dst"] = edges["dst"].map(zip2state)
    edges.loc[:, "src_id"] = edges["src"].map(ein2id)
    edges.loc[:, "dst_id"] = edges["dst"].map(ein2id)

    del ein2id, zip2state
    print("number of source region nodes:", len(edges[edges["src"].str.len() == 2]))
    print("number of dst region nodes:", len(edges[edges["dst"].str.len() == 2]))
    edges = (
        edges.copy()
        .dropna(axis=0, how="any", subset=["src_id", "dst_id"])
        .reset_index(drop=True)
    )
    print("number of source region nodes:", len(edges[edges["src"].str.len() == 2]))
    print("number of dst region nodes:", len(edges[edges["dst"].str.len() == 2]))
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
    )

    # edge_attr = torch.from_numpy(
    #     edges[["cash_grant_minmax", "tax_period_minmax"]].values
    # ).to(device)

    # Spltting out train+val from test
    nodes_trainval = torch.from_numpy(
        nodes[nodes["benchmark_status"] == "train"].index.values
    )
    nodes_test = torch.from_numpy(
        nodes[nodes["benchmark_status"] == "test"].index.values
    )
    # target_trainval = torch.from_numpy(
    #     nodes[nodes["benchmark_status"] == "train"][f"{target}_y"].values,
    # )

    nodes_train, nodes_valid = train_test_split(
        nodes_trainval, test_size=0.2, random_state=seed,
    )

    # Create train, validation, and test masks for data
    train_mask = torch.zeros(len(nodes), dtype=torch.bool)
    val_mask = torch.zeros(len(nodes), dtype=torch.bool)
    test_mask = torch.zeros(len(nodes), dtype=torch.bool)

    train_mask[nodes_train] = True
    val_mask[nodes_valid] = True
    test_mask[nodes_test] = True

    if hetero_graph:

        data = HeteroData()

        ##################
        # Add node types #
        ##################

        data["organization"].x = x[
            torch.from_numpy(nodes[nodes["node_type"] == "organization"].index.values)
        ]  # [num_grantors, num_features_grantors]
        data["organization"].y = y[
            torch.from_numpy(nodes[nodes["node_type"] == "organization"].index.values)
        ]  # [num_grantors, num_features_grantors]
        # Train mask
        data["organization"].train_mask = train_mask[
            torch.from_numpy(nodes[nodes["node_type"] == "organization"].index.values)
        ]

        # Valid mask
        data["organization"].val_mask = val_mask[
            torch.from_numpy(nodes[nodes["node_type"] == "organization"].index.values)
        ]
        # Test mask
        data["organization"].test_mask = test_mask[
            torch.from_numpy(nodes[nodes["node_type"] == "organization"].index.values)
        ]

        data["region"].x = x[
            torch.from_numpy(nodes[nodes["node_type"] == "region"].index.values)
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
            ),
        ]  # [2, num_edges_bothboth]

        data["organization", "operates_in", "region"].edge_index = edge_index[
            :,
            torch.from_numpy(
                edges[
                    (edges["src_node_type"] == "organization")
                    & (edges["dst_node_type"] == "region")
                ].index.values
            ),
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

        data = T.ToUndirected(merge=False)(data)
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
        data.val_mask = val_mask
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

