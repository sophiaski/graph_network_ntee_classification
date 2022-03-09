from utils import *

# Splitting data
from sklearn.model_selection import train_test_split

# from imblearn.over_sampling import ADASYN

# Model
from transformers import BertForSequenceClassification

# DataLoader
from torch.utils.data import DataLoader, WeightedRandomSampler, SequentialSampler


@typechecked
def stratify_none(
    X: np.ndarray, y: np.ndarray, seed: int = SEED
) -> ExperimentDataSplit:
    """_summary_

    Args:
        X (Sequence[str]): _description_
        y (Sequence[int]): _description_
        seed (int, optional): _description_. Defaults to SEED.

    Returns:
        List[Tuple[Sequence], Tuple[Sequence], Tuple[Sequence]]: _description_
    """
    # Split into train+val and test
    X_trainval, X_test, y_trainval, y_test = train_test_split(
        X, y, test_size=0.2, random_state=seed
    )

    # Split train into train-val
    X_train, X_validation, y_train, y_validation = train_test_split(
        X_trainval, y_trainval, test_size=0.1, random_state=seed
    )
    return {
        "train": (X_train, y_train),
        "validation": (X_validation, y_validation),
        "test": (X_test, y_test),
        "size": (len(y_train), len(y_validation), len(y_test)),
    }


@typechecked
def stratify_sklearn(
    X: np.ndarray, y: np.ndarray, seed: int = SEED
) -> ExperimentDataSplit:
    """_summary_

    Args:
        X (np.ndarray): _description_
        y (np.ndarray): _description_
        seed (int, optional): _description_. Defaults to SEED.

    Returns:
        ExperimentDataSplit: _description_
    """
    # Split into train+val and test
    X_trainval, X_test, y_trainval, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=seed
    )

    # Split train into train-val
    X_train, X_validation, y_train, y_validation = train_test_split(
        X_trainval, y_trainval, test_size=0.1, stratify=y_trainval, random_state=seed
    )
    return {
        "train": (X_train, y_train),
        "validation": (X_validation, y_validation),
        "test": (X_test, y_test),
        "size": (len(y_train), len(y_validation), len(y_test)),
    }


@typechecked
def get_dataloader(
    data_train: torch.tensor,
    class_weights_train: torch.tensor,
    data_validation: torch.tensor,
    data_test: torch.tensor,
    batch_size: int,
    is_test: bool = False,
) -> Union[DataLoader, Tuple[DataLoader, DataLoader]]:
    """_summary_

    Args:
        data_train (torch.tensor): _description_
        data_validation (torch.tensor): _description_
        data_test (torch.tensor): _description_
        batch_size (int): _description_
        is_test (bool, optional): _description_. Defaults to False.

    Returns:
        Union[DataLoader, Tuple[DataLoader, DataLoader]]: _description_
    """
    if is_test:
        return DataLoader(
            dataset=data_test,
            sampler=SequentialSampler(data_test),
            batch_size=batch_size,
            num_workers=4,
        )

    weighted_random_sampler = WeightedRandomSampler(
        weights=class_weights_train,
        num_samples=len(class_weights_train),
        replacement=True,
    )
    dataloader_train = DataLoader(
        dataset=data_train,
        sampler=weighted_random_sampler,
        batch_size=batch_size,
        num_workers=4,
    )
    dataloader_validation = DataLoader(
        dataset=data_validation, shuffle=False, batch_size=batch_size, num_workers=4
    )
    return dataloader_train, dataloader_validation


@typechecked
def get_model(num_labels: int, verbose: bool = True) -> BertForSequenceClassification:
    """_summary_

    Args:
        num_labels (int): _description_
        verbose (bool, optional): _description_. Defaults to True.

    Returns:
        BertForSequenceClassification: _description_
    """
    model = BertForSequenceClassification.from_pretrained(
        "bert-base-uncased",
        num_labels=num_labels,
        output_attentions=False,
        output_hidden_states=False,
    )

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


# @typechecked
# def stratify_ADASYN(
#     X: np.ndarray, y: np.ndarray, seed: int = SEED
# ) -> List[Tuple[np.ndarray, np.ndarray]]:
#     """_summary_

#     Args:
#         X (Sequence[str]): _description_
#         y (Sequence[int]): _description_
#         seed (int, optional): _description_. Defaults to SEED.

#     Returns:
#         List[Tuple[Sequence], Tuple[Sequence], Tuple[Sequence]]: _description_
#     """

#     # Split into train+val and test
#     X_trainval, X_test, y_trainval, y_test = train_test_split(
#         X, y, test_size=0.25, random_state=seed
#     )

#     # Split train into train-val
#     X_train, X_validation, y_train, y_validation = train_test_split(
#         X_trainval, y_trainval, test_size=0.1, random_state=seed
#     )
#     X_resampled, y_resampled = ADASYN(
#         sampling_strategy="minority", random_state=seed
#     ).fit_resample(X_train, y_train)
#     return [(X_resampled, y_resampled), (X_validation, y_validation), (X_test, y_test)]

# encoded_data = tokenizer.batch_encode_plus(
#     data.sequence.values,
#     add_special_tokens=True,
#     pad_to_max_length=True,
#     truncation="longest_first",
#     return_attention_mask=True,
#     return_tensors="pt",
# )
