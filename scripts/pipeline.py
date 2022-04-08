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

# Metrics
from torchmetrics import Accuracy, F1Score

# PyTorch Dataset
from dataset import NGODataset


@typechecked
def build_data(
    cat_type: str,
    strat_type: str,
    max_length: int,
    sampler: str,
    frac: float = 1.0,
    seed: int = SEED,
    verbose: bool = True,
) -> ExperimentData:
    """_summary_

    Args:
        cat_type (str): _description_
        strat_type (str): _description_
        max_length (int): _description_
        sampler (str): _description_
        frac (float, optional): _description_. Defaults to 1.0.
        seed (int, optional): _description_. Defaults to SEED.
        verbose (bool, optional): _description_. Defaults to True.

    Raises:
        ValueError: _description_
        ValueError: _description_
        ValueError: _description_

    Returns:
        ExperimentData: _description_
    """
    if cat_type not in ["both", "broad", "ntee"]:
        raise ValueError("cat_type must be 'both','broad', or 'ntee'.")
    if strat_type not in ["both", "none", "sklearn"]:
        raise ValueError("strat_type must be 'both','none', or 'sklearn'.")
    if sampler not in ["norm", "weighted_norm"]:
        raise ValueError("sampler must be 'norm' or 'weighted_norm'.")

    # Load data
    from benchmark_process import load_benchmark
    from benchmark_process import load_benchmark

    input_data = load_benchmark(
        merge_data=True, frac=frac, seed=seed, verbose=verbose
    ).fillna("")
    # Encode labels as torch-friendly integers
    encoded_data_dict = encode_targets(
        data=input_data, cat_type=cat_type, verbose=verbose
    )
    # Split into train/validation/test sets, based onf stratification strategy
    split_data_dict = split_data(
        experiment_dict=encoded_data_dict,
        strat_type=strat_type,
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
        print(f"\nAPPLYING LabelEncoder() TO DATA: {data.shape}")
    if cat_type == "broad":
        col_name = "broad_cat"
        unlabeled = "X"
        group2name = BROAD_CAT_NAME
    else:
        col_name = "NTEE1"
        unlabeled = "Z"
        group2name = NTEE_NAME

    # Remove unlabeled data then reset index (easier to combine embeddings after), sorted by train->test->new
    unlabeled_data = data[data[col_name] == unlabeled].copy()
    data = (
        data[data[col_name] != unlabeled]
        .copy()
        .sort_values("benchmark_status", ascending=False)
    )

    # Encode groups as numeric target value
    data.loc[:, f"{col_name}_target"] = preprocessing.LabelEncoder().fit_transform(
        data[col_name].values
    )
    # Create output mapper dicts
    target2group = data.groupby(f"{col_name}_target")[col_name].first().to_dict()

    # Drop original group values
    data.drop(["NTEE1", "broad_cat"], axis=1, inplace=True)
    if verbose:
        print(
            f"\tREPLACED CATEGORY NAME COLUMN W/ ENCODED TARGETS {data.columns.tolist()} -> ['sequence', 'target']"
        )

    # Reformat dataframe to only have input sequence & target label as columns
    subset_data = (
        data[["ein", "benchmark_status", "sequence", f"{col_name}_target"]]
        .copy()
        .rename({"broad_cat_target": "target", "NTEE1_target": "target"}, axis=1)
    )

    # Return output experiment dictionary
    return {
        "data": subset_data,
        "data_unlabeled": unlabeled_data,
        "num_labels": subset_data["target"].nunique(),
        "target2group": target2group,
        "group2name": group2name,
    }


@typechecked
def split_data(
    experiment_dict: ExperimentData,
    strat_type: str,
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
        print(f"\nSPLITTING DATA INTO TRAIN/DEV/TEST")
    data = experiment_dict["data"]

    if strat_type == "none":  # No stratified sampling
        experiment_dict["stratify_none"] = stratify_by(
            data=data, stratify=False, seed=seed
        )
    elif strat_type == "sklearn":  # Use stratified sampling
        experiment_dict["stratify_sklearn"] = stratify_by(
            data=data, stratify=True, seed=seed
        )
    if verbose:
        print(
            f"\Train/Valid/Test split size: {experiment_dict[f'stratify_{strat_type}']['split_size']}"
        )
    return experiment_dict


@typechecked
def stratify_by(
    data: pd.DataFrame, stratify: bool = True, seed: int = SEED,
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
    # Split datafranmes into train+val and test (based on benchmark)
    trainval = data[data["benchmark_status"] != "test"].copy()
    test = data[data["benchmark_status"] == "test"].copy()

    if stratify:
        # Split train into train-val w/ stratification
        (train, validation,) = train_test_split(
            trainval,
            test_size=0.3,
            stratify=trainval["target"].values,
            random_state=seed,
        )
    else:
        # Split train into train-val w/o stratifying
        (train, validation,) = train_test_split(
            trainval, test_size=0.3, random_state=seed,
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

    for strat_type in EXPERIMENT_KEYS[1]:
        if experiment_dict.get(f"stratify_{strat_type}"):
            if verbose:
                print(f"\tGiven a stratification strategy ({strat_type.upper()}...")
            # Encode train, validation, test
            for split in EXPERIMENT_KEYS[2]:
                if verbose:
                    print(f"\t\tCreate a DataSet ({split.upper()}).")
                # Extract (X,y) tuple
                split_df = experiment_dict[f"stratify_{strat_type}"][split]

                experiment_dict[f"stratify_{strat_type}"][
                    f"dataset_{split}"
                ] = NGODataset(dataframe=split_df, max_length=max_length)
                # Add class weights for training data
                if split == "train":
                    targets = split_df["target"].values

                    class_counts = torch.tensor(
                        [val[1] for val in sorted(Counter(targets).items())]
                    )
                    class_weights = 1.0 / class_counts
                    if sampler == "weighted_norm":
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
        dataset=data_train, sampler=weighted_random_sampler, batch_size=batch_size,
    )
    # Sequential sampling for validation
    dataloader_validation = DataLoader(
        dataset=data_validation, shuffle=False, batch_size=batch_size,
    )
    # Sequential sampling for test
    dataloader_test = DataLoader(
        dataset=data_test, shuffle=False, batch_size=batch_size,
    )
    return {
        "train": dataloader_train,
        "valid": dataloader_validation,
        "test": dataloader_test,
    }


@typechecked
def load_model(
    num_labels: int, classifier_dropout: float, verbose: bool = False
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
def train_epoch(
    model: BertForSequenceClassification,
    dataloader: DataLoader,
    optimizer: Union[optim.Adam, optim.SGD],
    scheduler: optim.lr_scheduler.LambdaLR,
    epoch: int,
    clip_grad: bool,
    device: torch.device,
) -> float:

    # Update state to train
    model.train()
    cumu_loss = 0.0

    # Set up progress bar
    progress_bar = tqdm(
        dataloader, desc=f"Epoch {epoch+1:1d}", leave=False, disable=False,
    )

    # Progress bar that iterates over training dataloader
    for batch in progress_bar:
        print(batch)
        # Sets gradients of all model parameters to zero.
        model.zero_grad(set_to_none=True)
        for b in batch:
            print(b)
            print(type(b))
        # Process batched data, save to device
        batch = tuple(b.to(device) for b in batch)
        inputs = {
            "input_ids": batch[0].to(device),
            "attention_mask": batch[1],
            "labels": batch[2],
        }

        # ➡ Forward pass
        outputs = model(**inputs)
        loss = outputs[0]
        cumu_loss += loss.item()

        # ⬅ Backward pass + weight update
        loss.backward()
        # Clips gradient norm of an iterable of parameters.
        # It is used to mitigate the problem of exploding gradients.
        if clip_grad:
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

        optimizer.step()

        # Taking the average, loss is indifferent to the batch size.
        loss_avg_batch = loss.item() / len(batch)

        # Log to W&B and print output
        wandb.log({"loss_batch": loss_avg_batch})
        progress_bar.set_postfix({"loss_batch": f"{loss_avg_batch:.3f}"})

        # Adjust the learning rate based on the number of batches
        scheduler.step()

    # Calculate average loss for entire training dataset.
    loss_avg = cumu_loss / len(dataloader)

    # Log to W&B and print output
    wandb.log({"loss": loss_avg})
    tqdm.write(f"loss: {loss_avg}")

    return loss_avg


@typechecked
def valid_epoch(
    model: BertForSequenceClassification,
    dataloader: DataLoader,
    epoch: int,
    num_labels: int,
    class_labels: Sequence[str],
    device: torch.device,
) -> float:

    # Update state to eval
    model.eval()
    cumu_loss = 0.0

    # Set up progress bar
    progress_bar = tqdm(
        dataloader, desc=f"Epoch {epoch+1:1d}", leave=False, disable=False,
    )

    # Create empty lists for collecting data to summarize later
    preds_all, true_labels_all, eins_all, embs_all = [], [], [], []

    # Initialize metrics
    accuracy_by_label = Accuracy(num_classes=num_labels, average="none")
    accuracy_weighted = Accuracy(num_classes=num_labels, average="weighted")

    # Precision: What proportion of predicted positives are truly positive?
    # Recall: what proportion of actual positives are correctly classified?
    # F1Score: harmonic mean of precision and recall
    f1score_by_label = F1Score(num_classes=num_labels, average="none")
    f1score_weighted = F1Score(num_classes=num_labels, average="weighted")

    # Progress bar that iterates over validation dataloader
    for batch in progress_bar:

        batch = tuple(b.to(device) for b in batch)

        inputs = {
            "input_ids": batch[0],
            "attention_mask": batch[1],
            "labels": batch[2],
            "eins": batch[3],
        }
        # torch.no_grad tells PyTorch not to construct the compute graph during this forward pass
        # (since we won’t be running backprop here)–this just reduces memory consumption and speeds things up a little.
        with torch.no_grad():
            outputs = model(**inputs)

        loss = outputs[0]
        cumu_loss += loss.item()
        #########################################
        # ACCUMULATE EMBEDDINGS AND MAP TO EINS #
        #########################################

        # Taking the average loss, loss is indifferent to the batch size.
        loss_avg_batch_valid = loss.item() / len(batch)

        # Calculate metrics on current batch
        preds = softmax(outputs[1].detach().cpu(), dim=-1)
        true_labels = inputs["labels"].cpu()
        acc = accuracy_weighted(preds, true_labels)
        acc_by_label = accuracy_by_label(preds, true_labels)
        f1 = f1score_weighted(preds, true_labels)
        f1_by_label = f1score_by_label(preds, true_labels)
        # auroc = auroc_weighted(preds, true_labels)

        # Log to W&B and print output
        wandb.log({"val_loss_batch": loss_avg_batch_valid})
        progress_bar.set_postfix({"val_loss_batch": f"{loss_avg_batch_valid:.3f}"})
        preds_all.append(preds)
        true_labels_all.append(true_labels)

    # Calculate average loss for entire validation dataset.
    loss_avg = cumu_loss / len(dataloader)

    # Convert lists to tensors
    true_labels_all = torch.cat(true_labels_all, dim=0)
    preds_all = torch.cat(preds_all, dim=0)
    preds_labels_all = preds_all.argmax(dim=-1)

    # Calculate metrics on all batches using custom accumulation.
    acc = accuracy_weighted.compute()
    acc_by_label = accuracy_by_label.compute()
    f1 = f1score_weighted.compute()
    f1_by_label = f1score_by_label.compute()

    # ROC curve
    # wandb.log({})
    # Confusion matrix
    # wandb.log({})
    # Log to W&B and print output
    epoch_scores = {"val_loss": loss_avg, "val_acc": acc, "val_f1": f1}
    epoch_scores.update(
        {f"val_acc_{idx}": val.item() for idx, val in enumerate(acc_by_label)}
    )
    epoch_scores.update(
        {f"val_f1_{idx}": val.item() for idx, val in enumerate(f1_by_label)}
    )
    wandb.log(epoch_scores)

    # Log PR curve, ROC curve, confusion matrix, all of the scores from this epoch
    wandb.log(
        {
            "pr_curve": wandb.plot.pr_curve(
                true_labels_all, preds_all, labels=class_labels,
            ),
            "roc_curve": wandb.plot.roc_curve(
                true_labels_all, preds_all, labels=class_labels,
            ),
            "conf_mat": wandb.plot.confusion_matrix(
                y_true=true_labels_all.numpy(),
                preds=preds_labels_all.numpy(),
                class_names=class_labels,
            ),
        },
    )
    # For progress display
    tqdm.write(f"val_loss: {loss_avg}")

    return loss_avg


def train(config, cat_type, strat_type, sampler, verbose=True):
    """
    Args:
        config (_type_): _description_
        cat_type (str): _description_
        strat_type (str): _description_
        sampler (str): _description_
        verbose (bool, optional): _description_
    """
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)

    # If there's a GPU available...
    if torch.cuda.is_available():
        # Tell PyTorch to use the GPU.
        device = torch.device("cuda")
        if verbose:
            print("There are %d GPU(s) available." % torch.cuda.device_count())
            print("We will use the GPU:", torch.cuda.get_device_name(0))
    # If not...
    else:
        if verbose:
            print("No GPU available, using the CPU instead.")
        device = torch.device("cpu")

    # Pass the hyperparameter values to wandb.init to populate wandb.config.
    with wandb.init(config=config):
        # If called by wandb.agent, as below,
        # this config will be set by Sweep Controller
        config = wandb.config

        # Load dataset
        data = build_data(
            cat_type=cat_type,
            strat_type=strat_type,
            max_length=config.max_length,
            sampler=sampler,
            frac=config.frac,
            seed=SEED,
            verbose=verbose,
        )
        num_labels = data["num_labels"]

        # Create mapper dicts
        target2group = data["target2group"]
        group2name = data["group2name"]
        target2name = {k: group2name[v] for k, v in target2group.items()}

        # Load BERT
        model = load_model(
            num_labels=num_labels,
            classifier_dropout=config.classifier_dropout,
            verbose=verbose,
        ).to(device)

        # Load dataloaders
        dataloader = get_dataloader(
            data_train=data[f"stratify_{strat_type}"]["dataset_train"],
            data_validation=data[f"stratify_{strat_type}"]["dataset_valid"],
            data_test=data[f"stratify_{strat_type}"]["dataset_test"],
            class_weights_train=data[f"stratify_{strat_type}"]["class_weights_train"],
            batch_size=config.batch_size,
        )

        # Optimizer
        optimizer = get_optimizer(
            model=model, learning_rate=config.learning_rate, optimizer=config.optimizer
        )

        # Scheduler
        scheduler = get_scheduler(
            optimizer=optimizer,
            num_warmup_steps=round(config.perc_warmup_steps * len(dataloader["train"])),
            num_batches=len(dataloader["train"]),
            epochs=config.epochs,
        )

        for epoch in range(config.epochs):

            if verbose:
                print(
                    f"Number of training examples: {len(dataloader['train'])*config.batch_size:,}"
                )

            # Train
            loss = train_epoch(
                model=model,
                dataloader=dataloader["train"],
                optimizer=optimizer,
                scheduler=scheduler,
                epoch=epoch,
                clip_grad=config.clip_grad,
                device=device,
            )

            # Validation
            val_loss = valid_epoch(
                model=model,
                dataloader=dataloader["valid"],
                epoch=epoch,
                num_labels=num_labels,
                class_labels=[target2name[lab] for lab in range(num_labels)],
                device=device,
            )

        # SAVE MODEL

        torch.cuda.empty_cache()

        return data


def train_broad_weighted_norm():
    """
    Experiment using
    ✓ Broad category targets for classification.
    ✓ Stratification in train/val/test split
    ✓ Normalizing the training class weights for weighted random sampling in the data loader
    """
    return train(
        config=SWEEP_INIT,
        cat_type="broad",
        strat_type="sklearn",
        sampler="weighted_norm",
    )


def train_broad_weighted():
    """
    Experiment using
    ✓ Broad category targets for classification.
    ✓ Stratification in train/val/test split
    ☓ NOT normalizing the training class weights for weighted random sampling in the data loader
    """
    return train(
        config=SWEEP_INIT, cat_type="broad", strat_type="sklearn", sampler="norm"
    )


def train_ntee_weighted_norm():
    """
    Experiment using
    ✓ NTEE1 category targets for classification.
    ✓ Stratification in train/val/test split
    ✓ Normalizing the training class weights for weighted random sampling in the data loader
    """
    return train(
        config=SWEEP_INIT,
        cat_type="ntee",
        strat_type="sklearn",
        sampler="weighted_norm",
    )


def train_ntee_weighted():
    """
    Experiment using
    ✓ NTEE1 category targets for classification.
    ✓ Stratification in train/val/test split
    ☓ NOT normalizing the training class weights for weighted random sampling in the data loader
    """
    return train(
        config=SWEEP_INIT, cat_type="ntee", strat_type="sklearn", sampler="norm"
    )
