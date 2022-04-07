# Initialize directory paths, classes, constants, packages, and other methods
from utils import *


from transformers import BertModel, BertTokenizer


def create_attribs(df, key_col, val_col):
    return pd.Series(df[val_col].values, index=df[key_col]).dropna().to_dict()


@typechecked
def load_graph_dataframes() -> Tuple[pd.DataFrame, pd.DataFrame]:
    # Load in data
    nodes = (
        pd.read_parquet(f"{GRANTS_GOLD_PATH}nodes.parquet")
        .replace("", pd.NA)
        .reset_index(drop=True)
    )
    edges = (
        pd.read_parquet(f"{GRANTS_GOLD_PATH}edges.parquet")
        .replace("", pd.NA)
        .reset_index(drop=True)
    )
    return nodes, edges[EDGE_COLS]


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


@typechecked
def create_graph() -> nx.Graph:

    nodes, edges = load_graph_dataframes()
    # Create edge tuple for networkx ID
    G = nx.from_pandas_edgelist(edges, source="source", target="destination")

    for col in ["cash_grant_amt", "edge_type", "grant_desc", "tax_period"]:
        nx.set_edge_attributes(G, create_attribs(edges, "edge", col), col)

    for col in ["NTEE1", "broad_cat", "locations", "sequence"]:
        nx.set_node_attributes(G, create_attribs(nodes, "ein", col), col)

    return G


@typechecked
def save_graph_to_json(G: nx.Graph) -> None:
    save_to_json(data=G, loc=GRAPH_GOLD_PATH, filename="test")


def main():
    create_graph()
    save_graph_to_json()


if __name__ == "__main__":
    main()
