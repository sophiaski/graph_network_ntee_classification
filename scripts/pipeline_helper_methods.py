"""All of the methods used with BERT Sequence Classification and hyperparameter tuning w/ W&B"""

# Initialize directory paths, classes, constants, packages, and other methods
from utils import *

# Preparing data
from sklearn import preprocessing

# Splitting data
from sklearn.model_selection import train_test_split

# DataLoader
from torch.utils.data import DataLoader, WeightedRandomSampler

# Model
from transformers import BertForSequenceClassification

# Optimizer
import torch.optim as optim

# Scheduler
from transformers import get_linear_schedule_with_warmup

# SoftMax, Normalize
from torch.nn.functional import softmax, normalize
from sklearn.utils import class_weight

# Metrics
from torchmetrics import Accuracy, F1Score

# PyTorch Dataset
from dataset import NGODataset

# Nodes
from graph_process import load_nodes


@typechecked
def build_data(
    cat_type: str,
    strat_type: str,
    max_length: int,
    sampler: str,
    split_size: float,
    complex_graph: bool = False,
    add_more_targets: bool = False,
    frac: float = 1.0,
    seed: int = SEED,
    verbose: bool = True,
) -> ExperimentData:
    """Set up data for the model pipeline!"""
    if cat_type not in ["broad", "ntee"]:
        raise ValueError("cat_type must be 'broad' or 'ntee'.")
    if strat_type not in ["none", "sklearn"]:
        raise ValueError("strat_type must be 'none' or 'sklearn'.")
    if sampler not in ["norm", "weighted_norm"]:
        raise ValueError("sampler must be 'norm' or 'weighted_norm'.")

    # Load data
    input_data = load_nodes(
        complex_graph=complex_graph,
        add_more_targets=add_more_targets,
        frac=frac,
        seed=seed,
        verbose=verbose,
    ).fillna("")

    # Encode labels as torch-friendly integers
    encoded_data_dict = encode_targets(
        data=input_data, cat_type=cat_type, verbose=verbose
    )
    # Split into train/validation/test sets, based onf stratification strategy
    split_data_dict = split_data(
        experiment_dict=encoded_data_dict,
        strat_type=strat_type,
        split_size=split_size,
        seed=seed,
        verbose=verbose,
    )
    # Tokenize and convert data into tensors
    experiment_dict = tokenize_data(
        experiment_dict=split_data_dict,
        max_length=max_length,
        sampler=sampler,
        verbose=verbose,
    )

    if verbose:
        print("Processing data is done. Now onto the model!")

    return experiment_dict


@typechecked
def encode_targets(
    data: pd.DataFrame, cat_type: str, verbose: bool = True,
) -> ExperimentData:
    """Encode target feature for both category types.

    Args:
        data (pd.DataFrame): Input data.
        cat_type (str): Category target feature.
        verbose (bool, optional): Debugging print statements. Defaults to True.

    Returns:
        ExperimentData: Custom TypedDict that contains all of the information needed for model training.
    """
    if verbose:
        print(f"Applying LabelEncoder() to target: {cat_type.upper()}")
    if cat_type == "broad":
        col_name = "broad_cat"
        unlabeled_group = "X"
        group2name = BROAD_CAT_NAME
    else:
        col_name = "NTEE1"
        unlabeled_group = "Z"
        group2name = NTEE_NAME

    # For unlabeled nodes, encode groups as -1
    unlabeled = data[data[col_name] == unlabeled_group].copy()
    unlabeled.loc[:, f"{col_name}_target"] = 0
    if verbose:
        print(
            f"\tFound {unlabeled.shape[0]} unlabeled organizations in this dataset, ignoring in training."
        )

    # Encode labeled groups as numeric target value
    data = data[data[col_name] != unlabeled_group].copy()
    data.loc[:, f"{col_name}_target"] = preprocessing.LabelEncoder().fit_transform(
        data[col_name].values
    )
    # Create output mapper dicts
    target2group = data.groupby(f"{col_name}_target")[col_name].first().to_dict()

    # Drop original group values
    data.drop(["NTEE1", "broad_cat"], axis=1, inplace=True)
    if verbose:
        print(
            f"\tReplaced columns with sequence & target columns \n\t\t{data.columns.tolist()}"
        )

    # Reformat dataframe to only have input sequence & target label as columns
    data = (
        data[["ein", "benchmark_status", "sequence", f"{col_name}_target"]]
        .copy()
        .rename({"broad_cat_target": "target", "NTEE1_target": "target"}, axis=1)
    )
    if verbose:
        print(f"\t\t{data.columns.tolist()}")
    # Return output experiment dictionary
    return {
        "data": data,
        "unlabeled": unlabeled[
            ["ein", "benchmark_status", "sequence", f"{col_name}_target"]
        ]
        .copy()
        .rename({"broad_cat_target": "target", "NTEE1_target": "target"}, axis=1),
        "num_labels": data["target"].nunique(),
        "target2group": target2group,
        "group2name": group2name,
    }


@typechecked
def split_data(
    experiment_dict: ExperimentData,
    strat_type: str,
    split_size: float,
    seed: int = SEED,
    verbose: bool = True,
) -> ExperimentData:
    """For each experiment, apply stratification method.

    Strategies:
        - None
        - Stratified sampling to ensure that relative class frequencies is approximately preserved in each train and validation fold
            Each set contains approximately the same percentage of samples of each target class as the complete set.

    See more here:
        https://scikit-learn.org/stable/modules/cross_validation.html#stratification

    Args:
        experiment_dict (ExperimentData): Custom TypedDict that contains all of the information needed for model training.
        strat_type (str): Stratification strategy.
        seed (int, optional): Random state for reproducibiltiy. Defaults to SEED.
        verbose (bool, optional): Debugging print statements. Defaults to True.

    Returns:
        ExperimentData: Custom TypedDict that contains all of the information needed for model training.
    """
    if verbose:
        print(f"Splitting data into train/valid/test.")
    # Load in data and split into labeled/nonlabeled examples
    data = experiment_dict["data"]

    if strat_type == "none":  # No stratified sampling
        if verbose:
            print("\tSplitting train+val by randomness.")
        experiment_dict["stratify_none"] = stratify_by(
            data=data, split_size=split_size, stratify=False, seed=seed
        )
    elif strat_type == "sklearn":  # Use stratified sampling
        if verbose:
            print("\tStratifiying train+val by target values.")
        experiment_dict["stratify_sklearn"] = stratify_by(
            data=data, split_size=split_size, stratify=True, seed=seed
        )

    if verbose:
        print(
            f"\tSplit sizes: {experiment_dict[f'stratify_{strat_type}']['split_size']}"
        )
    # Add unlabeled data back
    experiment_dict[f"stratify_{strat_type}"]["unlabeled"] = experiment_dict[
        "unlabeled"
    ]
    return experiment_dict


@typechecked
def stratify_by(
    data: pd.DataFrame, split_size: float, stratify: bool = True, seed: int = SEED,
) -> ExperimentDataSplit:
    """Split arrays into random train and test subsets.

    Read more in sklearn documentaion:
    https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.train_test_split.html

    Args:
        data (pd.DataFrame): 
        stratify (bool, optional): If True, data is split in a stratified fashion, using the class labels. If False, keep original benchmark split for test, stratify the train/val.  Defaults to True.
        seed (int, optional): Random state for reproducibiltiy. Defaults to SEED.

    Returns:
        ExperimentDataSplit: Return train, validation, and test data; and sizes of each.
    """
    # Split dataframes into train+val and test (based on benchmark)

    # Add new data intro train+val if it exists (add_more_targets == True)
    trainval = data[data["benchmark_status"] != "test"].copy()

    # Benchmark test stays the same
    test = data[data["benchmark_status"] == "test"].copy()

    if stratify:
        # Split train into train-val w/ stratification
        (train, validation,) = train_test_split(
            trainval,
            test_size=split_size,
            stratify=trainval["target"].values,
            random_state=seed,
        )
    else:
        # Split train into train-val w/o stratifying
        (train, validation,) = train_test_split(
            trainval, test_size=split_size, random_state=seed,
        )

    return {
        "train": train,
        "valid": validation,
        "test": test,
        "split_size": (len(train), len(validation), len(test)),
    }


@typechecked
def tokenize_data(
    experiment_dict: ExperimentData,
    max_length: int,
    sampler: str,
    verbose: bool = True,
) -> ExperimentData:
    """For a given experiment / stratify strategy, tokenzie the train, validation, and test sets and save them as tensors.

    Args:
        experiment_dict (ExperimentData): Custom TypedDict that contains all of the information needed for model training.
        max_length (int): Hyperparameter that controls the length of the padding/truncation for the tokenized text.
        sampler (str): Hyperparameter that specifies whether or not to normalize the class weights in the data preparation step.
        verbose (bool, optional): Debugging print statements. Defaults to True.

    Returns:
        ExperimentData: Custom TypedDict that contains all of the information needed for model training.
    """
    if verbose:
        print("Creating NGODatasets for dataloaders.")

    for strat_type in EXPERIMENT_KEYS[1]:
        if experiment_dict.get(f"stratify_{strat_type}"):
            if verbose:
                print(f"\tApplying a stratification strategy: {strat_type.upper()}.")
            # Encode train, validation, test
            for split in EXPERIMENT_KEYS[2]:
                if verbose:
                    print(f"\t\t - Create a NGODataset: {split.upper()}")
                # Extract (X,y) tuple
                split_df = experiment_dict[f"stratify_{strat_type}"][split]

                experiment_dict[f"stratify_{strat_type}"][
                    f"dataset_{split}"
                ] = NGODataset(dataframe=split_df, max_length=max_length)
                # Add class weights for training data
                if split == "train":
                    if verbose:
                        print(
                            f"\t\t\tCreating class weights to help with imbalance in WeightedRandomSampler."
                        )
                    # Converting target labels into tensor
                    targets = torch.tensor(split_df["target"].values, dtype=torch.long)
                    class_weights = torch.tensor(
                        class_weight.compute_class_weight(
                            "balanced", np.unique(targets), targets.numpy()
                        ),
                        dtype=torch.float,
                    )
                    if sampler == "weighted_norm":
                        if verbose:
                            print("\t\t\t (Normalize them)")
                        # https://discuss.pytorch.org/t/what-is-reasonable-range-of-per-class-weights/41742/5
                        class_weights = normalize(input=class_weights, dim=0)
                    experiment_dict[f"stratify_{strat_type}"][
                        f"class_weights_{split}"
                    ] = class_weights[targets]

    return experiment_dict


@typechecked
def get_dataloader(
    data_train: NGODataset,
    data_validation: NGODataset,
    data_test: NGODataset,
    data_unlabeled: NGODataset,
    class_weights_train: Tensor,
    batch_size: int,
) -> DataLoaderDict:
    """Convert the Dataset wrapping tensors into DataLoader objects to be fed into the model.

    Args:
        data_train (NGODataset): Training dataset.
        data_validation (NGODataset): Validation dataset.
        data_test (NGODataset): Test dataset.
        class_weights_train (Tensor): Class weights that are reciprocal of class counts to oversample minority classes.
        batch_size (int): Hyperparameter. Specified batch size for loading data into the model.

    Returns:
        DataLoaderDict: Returns either train/val DataLoaders or the test DataLoader, which are iterables over the given dataset(s).
    """

    # Sample the input based on the passed weights.
    weighted_random_sampler = WeightedRandomSampler(
        weights=class_weights_train, num_samples=len(class_weights_train)
    )
    dataloader_train = DataLoader(
        dataset=data_train,
        sampler=weighted_random_sampler,
        batch_size=batch_size,
        num_workers=4,
    )
    # Sequential sampling for validation
    dataloader_validation = DataLoader(
        dataset=data_validation, shuffle=False, batch_size=64,
    )
    # Sequential sampling for test
    dataloader_test = DataLoader(dataset=data_test, shuffle=False, batch_size=64)

    # Sequential sampling for unlabeled
    dataloader_unlabeled = DataLoader(
        dataset=data_unlabeled, shuffle=False, batch_size=64,
    )

    return {
        "train": dataloader_train,
        "valid": dataloader_validation,
        "test": dataloader_test,
        "unlabeled": dataloader_unlabeled,
    }


@typechecked
def load_model(
    num_labels: int, classifier_dropout: float, verbose: bool = False,
) -> BertForSequenceClassification:
    """Load the Bert Model transformer for sequence classification from Hugging Face.

    See more here:
    https://huggingface.co/docs/transformers/model_doc/bert

    Source code:
    https://github.com/huggingface/transformers/blob/v4.17.0/src/transformers/models/bert/modeling_bert.py#L1501

    Args:
        num_labels (int): Number of target labels in the dataset.
        classifier_dropout (int):
        verbose (bool, optional): Print summary of model parameters. Defaults to False.

    Returns:
        BertForSequenceClassification: Bert Model transformer with a sequence classification/regression head on top (a linear layer on top of the pooled output).
    """
    model = BertForSequenceClassification.from_pretrained(
        "bert-base-uncased",
        num_labels=num_labels,
        classifier_dropout=classifier_dropout,
        output_attentions=False,
        output_hidden_states=True,
    )
    # Summarize model
    if verbose:
        # Get all of the model's parameters as a list of tuples.
        params = list(model.named_parameters())
        print(
            "The BERT model has {:} different named parameters.\n".format(len(params))
        )
        print("\t==== Embedding Layer ====\n")
        for p in params[0:5]:
            print("\t{:<55} {:>12}".format(p[0], str(tuple(p[1].size()))))
        print("\n\t==== First Transformer ====\n")
        for p in params[5:21]:
            print("\t{:<55} {:>12}".format(p[0], str(tuple(p[1].size()))))
        print("\n\t==== Output Layer ====\n")
        for p in params[-4:]:
            print("\t{:<55} {:>12}".format(p[0], str(tuple(p[1].size()))))

    return model


@typechecked
def get_optimizer(
    model: BertForSequenceClassification, learning_rate: float, optimizer: str = "adam",
) -> Union[optim.Adam, optim.SGD]:
    """Optimizer that implements adam algoirthm or stochastic gradient descent.

    Args:
        model (BertForSequenceClassification): BertForSequenceClassification: Bert Model transformer with a sequence classification/regression head on top (a linear layer on top of the pooled output).
        lr (float): Hyperparameter. Learning rate.
        optim_type (str, optional): Hyperparameter. Type of optimizer to output. Defaults to "adam".

    Returns:
        Union[optim.Adam, optim.SGD]: Optimizer
    """
    if optimizer == "adam":
        return optim.Adam(params=model.parameters(), lr=learning_rate)
    elif optimizer == "sgd":
        return optim.SGD(params=model.parameters(), lr=learning_rate)


@typechecked
def get_scheduler(
    optimizer: Union[optim.Adam, optim.SGD],
    num_warmup_steps: int,
    num_batches: int,
    epochs: int,
) -> optim.lr_scheduler.LambdaLR:
    """
    Linear Warmup is a learning rate schedule where we linearly increase the learning rate from a low rate to a constant rate thereafter.
    This reduces volatility in the early stages of training.

    It increases the learning rate from 0 to initial_lr specified in your optimizer in num_warmup_steps, after which it becomes constant.

    See more here:
    https://huggingface.co/transformers/v3.5.1/_modules/transformers/optimization.html

    Args:
        optimizer (Union[optim.Adam, optim.SGD]): The optimizer for which to schedule the learning rate.
        num_warmup_steps (int): The number of steps for the warmup phase.
        num_batches (int): Number of batches in the training dataset.
        epochs (int): Number of passes through the dataset.

    Returns:
        optim.lr_scheduler.LambdaLR: HuggingFace scheduler.
    """
    # Total number of training steps (num_training_step) is [number of batches] x [number of epochs].
    return get_linear_schedule_with_warmup(
        optimizer=optimizer,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=num_batches * epochs,
    )


@typechecked
def create_saved_model_path(
    ntee: bool = False,
    complex_graph: bool = False,
    add_more_targets: bool = False,
    saved_model_vers: Optional[str] = None,
) -> str:
    filepath = MODELS_PATH
    if ntee:
        filepath += "ntee"
    else:
        filepath += "broad"
    if complex_graph:
        filepath += "_complex"
    else:
        filepath += "_simple"
    if add_more_targets:
        filepath += "_w_more_targets"
    if saved_model_vers:
        filepath += f"_{saved_model_vers}"
    return filepath


@typechecked
def create_CONFIG(
    ntee: bool = False,
    complex_graph: bool = False,
    add_more_targets: bool = False,
    from_saved_model: bool = False,
    saved_model_vers: Optional[str] = None,
) -> Dict:
    ALL_CONFIG = {
        "optimizer": "adam",
        "epochs": 3,
        "batch_size": 32,
        "max_length": 128,
        "clip_grad": False,
        "split_size": 0.2,
        "frac": 1.0,
        "strat_type": "sklearn",
        "sampler": "weighted_norm",
    }

    if ntee:
        # Best model: NTEE
        CATEGORY_CONFIG = {
            "cat_type": "ntee",
            "learning_rate": 3e-05,
            "classifier_dropout": 0.4,
            "perc_warmup_steps": 0,
        }
    else:
        # Best model: BROAD
        CATEGORY_CONFIG = {
            "cat_type": "broad",
            "learning_rate": 5e-05,
            "classifier_dropout": 0.3,
            "perc_warmup_steps": 0.1,
        }
    Merge(ALL_CONFIG, CATEGORY_CONFIG)

    if add_more_targets:
        ALL_CONFIG["add_more_targets"] = True
    else:
        ALL_CONFIG["add_more_targets"] = False
    if complex_graph:
        ALL_CONFIG["complex_graph"] = True
    else:
        ALL_CONFIG["complex_graph"] = False

    if from_saved_model:
        ALL_CONFIG["saved_model_path"] = create_saved_model_path(
            ntee=ntee,
            complex_graph=complex_graph,
            add_more_targets=add_more_targets,
            saved_model_vers=saved_model_vers,
        )
    else:
        ALL_CONFIG["saved_model_path"] = None
    return ALL_CONFIG


NTEE_SIMPLE = create_CONFIG(ntee=True, complex_graph=False)
NTEE_COMPLEX = create_CONFIG(ntee=True, complex_graph=True)
BROAD_SIMPLE = create_CONFIG(ntee=False, complex_graph=False)
BROAD_COMPLEX = create_CONFIG(ntee=False, complex_graph=True)
NTEE_SIMPLE_SAVED = create_CONFIG(
    ntee=True, complex_graph=False, from_saved_model=True, saved_model_vers="04-09"
)
NTEE_COMPLEX_SAVED = create_CONFIG(
    ntee=True, complex_graph=True, from_saved_model=True, saved_model_vers="04-09"
)
BROAD_SIMPLE_SAVED = create_CONFIG(
    ntee=False, complex_graph=False, from_saved_model=True, saved_model_vers="04-09"
)
BROAD_COMPLEX_SAVED = create_CONFIG(
    ntee=False, complex_graph=True, from_saved_model=True, saved_model_vers="04-09"
)

