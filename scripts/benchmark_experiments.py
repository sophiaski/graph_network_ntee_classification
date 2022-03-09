# Initialize directory paths, classes, constants, packages, and other methods
from utils import *

# Model methods for treating the data
from model_methods import *

# Tokenizer
from joblib import Parallel, delayed
from transformers import BatchEncoding, BertTokenizer

# Preparing data
from sklearn import preprocessing
from collections import Counter
from torch.utils.data import TensorDataset

# Weights and Biases
import wandb

@typechecked
def load_data(
    frac: float = 1.0,
    seed: int = SEED,
    verbose: bool = True,
) -> pd.DataFrame:
    """Load in the data, currently stored as 2 separate files.

    Args:
        frac (float, optional): Return a random fraction of rows from Pandas DataFrame. Defaults to 1.0 (100%).
        seed (int, optional): Random state for reproducibiltiy. Defaults to 42.
        verbose (bool, optional): Print Pandas DataFrame shape. Defaults to True.

    Returns:
        pd.DataFrame: Combined dataset.
    """
    df_list = []
    for val in ["train", "test"]:
        filepath = f"{BENCHMARK_MODEL}df_ucf_{val}.parquet"
        df_list.append(pd.read_parquet(filepath))
    data = (
        pd.concat(df_list)
        .sample(frac=frac, random_state=seed)
        .rename({"input": "sequence"}, axis=1)
        .reset_index(drop=True)
    )
    if verbose:
        print(f"SAMPLING {round(frac*100,2)}% OF DATA...")
        print(f"\tDATA SIZE: {data.shape}")
    return data


@typechecked
def encode_targets(
    data: pd.DataFrame,
    verbose: bool = True,
) -> ExperimentDict:
    """Encode target feature.

    Args:
        data (pd.DataFrame): Input data.
        verbose (bool, optional): Print Pandas DataFrame shape. Defaults to True.

    Returns:
        ExperimentDict: Return the experiment dictionary with target feature dataset and a mappers to convert from target back to the original group.
    """
    if verbose:
        print(f"\nAPPLYING LabelEncoder() TO DATA: {data.shape}")

    # Encode groups as numeric target value
    data["broad_cat_target"] = preprocessing.LabelEncoder().fit_transform(
        data["broad_cat"].values
    )
    data["NTEE1_target"] = preprocessing.LabelEncoder().fit_transform(
        data["NTEE1"].values
    )

    # Create output mapper dicts
    broad_target2group = data.groupby("broad_cat_target")["broad_cat"].first().to_dict()
    ntee_target2group = data.groupby("NTEE1_target")["NTEE1"].first().to_dict()

    # Drop original group values
    data.drop(["NTEE1", "broad_cat"], axis=1, inplace=True)
    if verbose:
        print(f"\tREPLACED CATEGORY BY ENCODED TARGET COLUMNS: {data.columns.tolist()}")

    # Create output experiment dictionary
    subset_broad = (
        data[["sequence", "broad_cat_target"]]
        .copy()
        .rename({"broad_cat_target": "target", "NTEE1_target": "target"}, axis=1)
    )
    subset_ntee = (
        data[["sequence", "NTEE1_target"]]
        .copy()
        .rename({"broad_cat_target": "target", "NTEE1_target": "target"}, axis=1)
    )
    return {
        "broad": {
            "data": subset_broad,
            "target2group": broad_target2group,
            "group2name": BROAD_CAT_NAME,
        },
        "ntee": {
            "data": subset_ntee,
            "target2group": ntee_target2group,
            "group2name": NTEE_NAME,
        },
    }


@typechecked
def split_data(
    experiment_dict: ExperimentDict,
    seed: int = SEED,
    verbose: bool = True,
) -> ExperimentDict:
    """

    3 Methods of sampling for each encoded target data set.
    - None
    - Stratified sampling to ensure that relative class frequencies is approximately preserved in each train and validation fold
        Each set contains approximately the same percentage of samples of each target class as the complete set.
        https://scikit-learn.org/stable/modules/cross_validation.html#stratification
    - Adaptive Synthetic (ADASYN) oversampling method, which was selected by the benchmark paper
        ADASYN generates new samples in training set by interpolation
        https://imbalanced-learn.org/stable/over_sampling.html# # ADASYN Documentation
        MODIFY EMBEDDING AND RECREATE BERT https://neptune.ai/blog/how-to-code-bert-using-pytorch-tutorial

    Args:
        data (pd.DataFrame): _description_
        seed (int, optional): _description_. Defaults to SEED.
        verbose (bool, optional): _description_. Defaults to True.

    Returns:
        ExperimentDict: _description_
    """
    if verbose:
        print(f"\nSPLITTING EXPERIMENTS INTO TRAIN/DEV/TEST")

    for cat in EXPERIMENT_KEYS[0]:
        df = experiment_dict[cat]["data"]
        X = df.loc[:, "sequence"].to_numpy()
        y = df.loc[:, "target"].to_numpy()
        experiment_dict[cat]["stratify_none"] = stratify_none(X=X, y=y, seed=seed)
        experiment_dict[cat]["stratify_sklearn"] = stratify_sklearn(X=X, y=y, seed=seed)
        # experiment_dict[cat]["stratify_ADASYN"] = stratify_none(X=X, y=y, seed=seed)
        if verbose:
            print(
                f"\t{cat.upper()} EXPERIMENT TRAIN/DEV/TEST: {experiment_dict[cat]['stratify_sklearn']['size']}"
            )
    return experiment_dict


@typechecked
def tokenize_data(
    tokenizer: BertTokenizer, experiment_dict: ExperimentDict, verbose: bool = True
) -> ExperimentDict:
    """_summary_

    Args:
        tokenizer (BertTokenizer): _description_
        experiment_dict (ExperimentDict): _description_
        verbose (bool, optional): _description_. Defaults to True.

    Returns:
        ExperimentDict: _description_
    """
    global func_encode_string

    def func_encode_string(sequence: str) -> BatchEncoding:
        return tokenizer.encode_plus(
            sequence,
            add_special_tokens=True,
            truncation="longest_first",
            padding="max_length",
            return_attention_mask=True,
            return_tensors="pt",
        )

    if verbose:
        verbose_num = 1
    else:
        verbose_num = 0
    for cat in EXPERIMENT_KEYS[0]:
        print(f"\n{cat.upper()} EXPERIMENT")
        for strat in EXPERIMENT_KEYS[1]:
            print(f"\t{strat.upper()} TOKENIZING...")
            for split in EXPERIMENT_KEYS[2]:
                print(f"\t\t{split.upper()}")
                # Extract (X,y) tuple
                split_tuple = experiment_dict[cat][strat][split]
                sequences = split_tuple[0]
                targets = split_tuple[1]
                # Loop over sequences create tokenize ids and attention masks
                input_ids = []
                attention_masks = []
                encoded_outputs = Parallel(
                    n_jobs=-1,
                    backend="multiprocessing",
                    batch_size="auto",
                    verbose=verbose_num,
                )(delayed(func_encode_string)(seq) for seq in sequences)
                for encoded_output in encoded_outputs:
                    # Add the encoded sentence to the list.
                    input_ids.append(encoded_output["input_ids"])
                    # And its attention mask (simply differentiates padding from non-padding).
                    attention_masks.append(encoded_output["attention_mask"])
                # Convert the lists into tensors.
                labels = torch.tensor(targets)
                input_ids = torch.cat(input_ids, dim=0)
                attention_masks = torch.cat(attention_masks, dim=0)
                experiment_dict[cat][strat][f"{split}_tensor"] = TensorDataset(
                    input_ids, attention_masks, labels
                )
                # Add class weights for training data
                if split == "train":
                    class_weights = 1.0 / torch.tensor(
                        [val[1] for val in sorted(Counter(targets).items())]
                    )

                    experiment_dict[cat][strat][
                        f"{split}_class_weights"
                    ] = class_weights[targets]
    return experiment_dict


def main(
    input_data: pd.DataFrame,
    seed: int = SEED,
    verbose: bool = True,
):
    """
    Initialize experiments.
    """
    print("\n======== Encode & Tokenize Data ================")
    # Begin timer.
    t0 = time.time()
    encoded_data_dict = encode_targets(data=input_data, verbose=verbose)
    split_data_dict = split_data(
        experiment_dict=encoded_data_dict, seed=seed, verbose=verbose
    )
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    experiment_dict = tokenize_data(
        tokenizer=tokenizer, experiment_dict=split_data_dict, verbose=verbose
    )
    time.sleep(0.5)
    # Calculate elapsed time in minutes.
    t0 = format_time(time.time() - t0)
    # Report progress.
    print(f"Elapsed: {t0} minutes")
    #####################################################
    #####################################################
    #####################################################
    print("\n================ Experiment Loops ================\n")
    wandb.login()
    # If there's a GPU available...
    if torch.cuda.is_available():
        # Tell PyTorch to use the GPU.
        device = torch.device("cuda")
        print("There are %d GPU(s) available." % torch.cuda.device_count())
        print("We will use the GPU:", torch.cuda.get_device_name(0))
    # If not...
    else:
        print("No GPU available, using the CPU instead.")
        device = torch.device("cpu")
    # Begin!
    for cat in EXPERIMENT_KEYS[0]:
        num_labels = experiment_dict[cat]["num_labels"]
        for strat in EXPERIMENT_KEYS[1]:
            model_name = f"(Target: {cat.upper()}, Stratify: {strat.upper()})"
            print(model_name)
            # Load model
            model = get_model(num_labels=num_labels, verbose=False)
            model.to(device)
            # Dataloaders
            get_dataloader(
                data_train=experiment_dict[cat][strat]["train_tensor"],
                class_weights_train=experiment_dict[cat][strat]["train_class_weights"],
                data_validation=experiment_dict[cat][strat]["validation_tensor"],
                batch_size=
            )
            
    # Begin timer.
    t1 = time.time()
    # https://towardsdatascience.com/pytorch-tabular-multiclass-classification-9f8211a123ab
    # WEIGHTED RANDOM SAMPLER
    # Calculate elapsed time in minutes.
    t1 = format_time(time.time() - t1)
    # Report progress.
    print(f"Elapsed: {t1} minutes")
    #####################################################
    #####################################################
    #####################################################
    print("\n================ Model Config ================")
    # Begin timer.
    t2 = time.time()
    # Calculate elapsed time in minutes.
    t2 = format_time(time.time() - t2)
    # Report progress.
    print(f"Elapsed: {t2} minutes")
    return tokenized_data_dict


# if __name__ == "__main__":
#     main(train_frac=TEST_SAMPLE, seed=SEED)

# encoded_data = tokenizer.batch_encode_plus(
#     data.sequence.values,
#     add_special_tokens=True,
#     pad_to_max_length=True,
#     truncation="longest_first",
#     return_attention_mask=True,
#     return_tensors="pt",
# )
# # Convert the lists into tensors.
# input_ids = torch.cat(input_ids, dim=0)
# attention_masks = torch.cat(attention_masks, dim=0)
