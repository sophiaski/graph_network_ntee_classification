"""All of the methods used with BERT Sequence Classification and hyperparameter tuning w/ W&B"""

# π±π΄ππ π΅πΎπ ππ΄πππ΄π½π²π΄ π²π»π°πππΈπ΅πΈπ²π°ππΈπΎπ½

# Helper methods to clear up the script
from pipeline_helper_methods import *

# Metrics
from torchmetrics import Accuracy, F1Score, Precision, Recall
from torch.nn.functional import softmax

# BOW Embeddings
from sklearn.feature_extraction.text import CountVectorizer

# WandB
import wandb

# BERT model specs -- NO FINE-TUNING
# Major group
NTEE_SIMPLE = create_CONFIG(ntee=True, complex_graph=False)
NTEE_COMPLEX = create_CONFIG(ntee=True, complex_graph=True)
# Broad category
BROAD_SIMPLE = create_CONFIG(ntee=False, complex_graph=False)
BROAD_COMPLEX = create_CONFIG(ntee=False, complex_graph=True)

# BERT model specs -- FINE TUNED
# Major group
NTEE_SIMPLE_SAVED = create_CONFIG(
    ntee=True, complex_graph=False, from_saved_model=True, saved_model_vers="04-10"
)
NTEE_COMPLEX_SAVED = create_CONFIG(
    ntee=True, complex_graph=True, from_saved_model=True, saved_model_vers="04-10"
)
# Broad category
BROAD_SIMPLE_SAVED = create_CONFIG(
    ntee=False, complex_graph=False, from_saved_model=True, saved_model_vers="04-09"
)
BROAD_COMPLEX_SAVED = create_CONFIG(
    ntee=False, complex_graph=True, from_saved_model=True, saved_model_vers="04-10"
)


def train_broad():
    """
    Experiment using
    β Broad category targets for classification.
    β Stratification in train/val/test split
    β Normalizing the training class weights for weighted random sampling in the data loader
    """
    print(BROAD_COMPLEX)
    return train(
        config=BROAD_COMPLEX,
        cat_type="broad",
        strat_type="sklearn",
        sampler="weighted_norm",
        save_model=True,
    )


def train_ntee():
    """
    Experiment using
    β NTEE1 category targets for classification.
    β Stratification in train/val/test split
    β Normalizing the training class weights for weighted random sampling in the data loader
    """
    print(NTEE_SIMPLE)
    return train(
        config=NTEE_SIMPLE,
        cat_type="ntee",
        strat_type="sklearn",
        sampler="weighted_norm",
        save_model=True,
    )


# βββββββββββββββββββββββββββββββββββββββββββββββββββββββ
# βββββββββββββββββββββββββββββββββββββββββββββββββββββββ
# βββββββββββββββββββββββββββββββββββββββββββββββββββββββ
# βββββββββββββββββββββββββββββββββββββββββββββββββββββββ
# βββββββββββββββββββββββββββββββββββββββββββββββββββββββ
# βββββββββββββββββββββββββββββββββββββββββββββββββββββββ


def train(
    config: Mapping,
    cat_type: str,
    strat_type: str,
    sampler: str,
    verbose: bool = True,
    save_model: bool = False,
):
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
            split_size=config.split_size,
            complex_graph=config.complex_graph,
            add_more_targets=config.add_more_targets,
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
            data_unlabeled=data[f"stratify_{strat_type}"]["dataset_unlabeled"],
            class_weights_train=data[f"stratify_{strat_type}"]["class_weights_train"],
            batch_size=config.batch_size,
            save_emb=False,
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
                    f"Number of training examples: {len(dataloader['train'])*config.batch_size:,}\nNumber of batches: {len(dataloader['train']):,}"
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
                phase="val",
                from_saved_model=False,
                logging_to_wandb=True,
                save_emb=False,
            )

        # Save model
        if save_model:
            # Vars
            if cat_type == "ntee":
                ntee = True
            else:
                ntee = False
            filepath = create_saved_model_path(
                ntee=ntee,
                complex_graph=config.complex_graph,
                add_more_targets=config.add_more_targets,
                saved_model_vers=date.today().strftime("%m-%d"),
            )
            #  For exporting the model and running inference later.
            torch.save(
                model.state_dict(), filepath,
            )

        torch.cuda.empty_cache()

        return data


# ββββββββββββββββββββββββββββββββββββββββββββββββββββββββββ
# ββββββββββββββββββββββββββββββββββββββββββββββββββββββββββ
# ββββββββββββββββββββββββββββββββββββββββββββββββββββββββββ
# ββββββββββββββββββββββββββββββββββββββββββββββββββββββββββ
# ββββββββββββββββββββββββββββββββββββββββββββββββββββββββββ
# ββββββββββββββββββββββββββββββββββββββββββββββββββββββββββ


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

    # Update state to train, being accumulator for loss and counter for tracking # of WandB logs
    model.train()
    cumu_loss, log_cnt = 0.0, 0

    # Set up progress bar
    progress_bar = tqdm(
        dataloader, desc=f"Epoch {epoch+1:1d}", leave=False, disable=False,
    )

    # Progress bar that iterates over training dataloader
    for batch in progress_bar:
        # Sets gradients of all model parameters to zero.
        model.zero_grad(set_to_none=True)

        # Process batched data, save to device
        batch = tuple(
            v.to(device) for _, v in batch.items() if isinstance(v, torch.Tensor)
        )
        inputs = {
            "input_ids": batch[0].squeeze(1),
            "attention_mask": batch[2].squeeze(1),
            "labels": batch[3],
        }

        # β‘ Forward pass
        outputs = model(**inputs)
        loss = outputs[0]
        cumu_loss += loss.item()

        # β¬ Backward pass + weight update
        loss.backward()
        # Clips gradient norm of an iterable of parameters.
        # It is used to mitigate the problem of exploding gradients.
        if clip_grad:
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

        optimizer.step()

        # Taking the average, loss is indifferent to the batch size.
        loss_avg_batch = loss.item() / len(batch)

        # Log to W&B and print output
        if log_cnt % 10 == 0:
            wandb.log({"loss_batch": loss_avg_batch})
        log_cnt += 1
        progress_bar.set_postfix({"loss_batch": f"{loss_avg_batch:.3f}"})

        # Adjust the learning rate based on the number of batches
        scheduler.step()

    # Calculate average loss for entire training dataset.
    loss_avg = cumu_loss / len(dataloader)

    # Log to W&B and print output
    wandb.log({"loss": loss_avg})
    log_cnt += 1
    tqdm.write(f"loss: {loss_avg}")

    return loss_avg


# βββββββββββββββββββββββββββββββββββββββββββββββββββββββββββββββββββββββββ
# βββββββββββββββββββββββββββββββββββββββββββββββββββββββββββββββββββββββββ
# βββββββββββββββββββββββββββββββββββββββββββββββββββββββββββββββββββββββββ
# βββββββββββββββββββββββββββββββββββββββββββββββββββββββββββββββββββββββββ
# βββββββββββββββββββββββββββββββββββββββββββββββββββββββββββββββββββββββββ
# βββββββββββββββββββββββββββββββββββββββββββββββββββββββββββββββββββββββββ


@typechecked
def valid_epoch(
    model: BertForSequenceClassification,
    dataloader: DataLoader,
    epoch: int,
    num_labels: int,
    class_labels: Sequence[str],
    device: torch.device,
    phase: str,
    from_saved_model: bool = False,
    logging_to_wandb: bool = True,
    save_emb: bool = False,
) -> float:

    # Update state to eval
    model.eval()
    cumu_loss = 0.0

    if logging_to_wandb:
        log_cnt = 0

    # Set up progress bar
    progress_bar = tqdm(
        dataloader, desc=f"Epoch {epoch+1:1d}", leave=False, disable=False,
    )

    # Create empty lists for collecting data to summarize later
    preds_all, true_labels_all, eins_all, embs_all = [], [], np.array([]), []

    # Initialize metrics
    accuracy_by_label = Accuracy(num_classes=num_labels, average="none")
    accuracy_weighted = Accuracy(num_classes=num_labels, average="weighted")
    accuracy_macro = Accuracy(num_classes=num_labels, average="macro")

    # Precision: What proportion of predicted positives are truly positive?
    precision_weighted = Precision(num_classes=num_labels, average="weighted")
    precision_macro = Precision(num_classes=num_labels, average="macro")
    # Recall: what proportion of actual positives are correctly classified?
    recall_weighted = Recall(num_classes=num_labels, average="weighted")
    recall_macro = Recall(num_classes=num_labels, average="macro")

    # F1Score: harmonic mean of precision and recall
    f1score_by_label = F1Score(num_classes=num_labels, average="none")
    f1score_weighted = F1Score(num_classes=num_labels, average="weighted")
    f1score_macro = F1Score(num_classes=num_labels, average="macro")

    # Progress bar that iterates over validation dataloader
    for batch in progress_bar:

        # Accumulate EINs
        eins = np.array(batch["eins"])
        eins_all = np.append(eins_all, eins)

        # Process batched data, save to device
        batch = tuple(
            v.to(device) for _, v in batch.items() if isinstance(v, torch.Tensor)
        )
        inputs = {
            "input_ids": batch[0].squeeze(1),
            "attention_mask": batch[2].squeeze(1),
            "labels": batch[3],
        }

        # torch.no_grad tells PyTorch not to construct the compute graph during this forward pass
        # (since we wonβt be running backprop here)βthis just reduces memory consumption and speeds things up a little.
        with torch.no_grad():
            outputs = model(**inputs)

        loss = outputs[0]
        cumu_loss += loss.item()

        # Accumulate embeddings
        CLS_token = outputs[2][-1][:, 0].detach().cpu()
        embs_all.append(
            CLS_token
        )  # CLS token hidden state at 13th layer (before dense pooling and tanh)

        # Taking the average loss, loss is indifferent to the batch size.
        loss_avg_batch_valid = loss.item() / len(batch)

        # Calculate metrics on current batch
        preds = softmax(outputs[1].detach().cpu(), dim=-1)
        true_labels = inputs["labels"].cpu()

        # auroc = auroc_weighted(preds, true_labels)

        # Accuracy
        acc = accuracy_weighted(preds, true_labels)
        acc_by_label = accuracy_by_label(preds, true_labels)
        acc_macro = accuracy_macro(preds, true_labels)

        # F1 Score
        f1 = f1score_weighted(preds, true_labels)
        f1_by_label = f1score_by_label(preds, true_labels)
        f1_macro = f1score_macro(preds, true_labels)

        # Precision
        prec = precision_weighted(preds, true_labels)
        prec_macro = precision_macro(preds, true_labels)

        # Recall
        rec = recall_weighted(preds, true_labels)
        rec_macro = recall_macro(preds, true_labels)

        # Log to W&B and print output
        if logging_to_wandb:
            if log_cnt % 10 == 0:
                wandb.log({f"{phase}_loss_batch": loss_avg_batch_valid})
                log_cnt += 1
        progress_bar.set_postfix({f"{phase}_loss_batch": f"{loss_avg_batch_valid:.3f}"})

        # Collect predictions each batch
        preds_all.append(preds)
        true_labels_all.append(true_labels)

    # Calculate average loss for entire dataset
    loss_avg = cumu_loss / len(dataloader)

    # Convert lists to tensors
    true_labels_all = torch.cat(true_labels_all, dim=0)
    preds_all = torch.cat(preds_all, dim=0)
    preds_labels_all = preds_all.argmax(dim=-1)
    embs_all = torch.cat(embs_all, dim=0).numpy()

    # Calculate metrics on all batches using custom accumulation.
    # Accuracy
    acc = accuracy_weighted.compute()
    acc_by_label = accuracy_by_label.compute()
    acc_macro = accuracy_macro.compute()

    # F1 score
    f1 = f1score_weighted.compute()
    f1_by_label = f1score_by_label.compute()
    f1_macro = f1score_macro.compute()

    # Precision
    prec = precision_weighted.compute()
    prec_macro = precision_macro.compute()

    # Recall
    rec = recall_weighted.compute()
    rec_macro = recall_macro.compute()

    # Save CLS token embeddings to numpy to merge with graph data later
    if save_emb:
        if from_saved_model:
            filename = f"{phase}_from_saved"
        else:
            filename = phase
        save_embeddings(
            eins_all, loc=EMBEDDINGS_PATH, filename=f"{filename}_eins", verbose=True
        )
        save_embeddings(
            embs_all, loc=EMBEDDINGS_PATH, filename=f"{filename}_embs", verbose=True
        )

    # Log all metrics WandB
    if logging_to_wandb:
        epoch_scores = {
            f"{phase}_loss": loss_avg,
            f"{phase}_acc": acc,
            f"{phase}_acc_macro": acc_macro,
            f"{phase}_f1": f1,
            f"{phase}_f1_macro": f1_macro,
            f"{phase}_prec": prec,
            f"{phase}_prec_macro": prec_macro,
            f"{phase}_rec": rec,
            f"{phase}_rec_macro": rec_macro,
        }
        epoch_scores.update(
            {f"{phase}_acc_{idx}": val.item() for idx, val in enumerate(acc_by_label)}
        )
        epoch_scores.update(
            {f"{phase}_f1_{idx}": val.item() for idx, val in enumerate(f1_by_label)}
        )
        wandb.log(epoch_scores)
        log_cnt += 1

        # Gather datapoints for plot
        if phase in ["train", "new"]:

            # Randomly sample 10K datapoints (WandB limit)
            sampling_indices_by_lab = {}
            min_idx = round(9999 / num_labels)

            # First grab indices for each label class in the true_labels_all tensor
            for lab in np.arange(num_labels):
                indices = (true_labels_all == lab).nonzero(as_tuple=True)[0]
                if len(indices) < min_idx:
                    min_idx = len(indices)
                sampling_indices_by_lab[lab] = indices

            # Then sample evenly across
            samps = []
            for lab in np.arange(num_labels):
                samps.append(
                    torch.tensor(
                        np.random.choice(
                            sampling_indices_by_lab[lab], size=min_idx, replace=False
                        )
                    )
                )
            # Combine to apply to the true labels / pred labels tensors
            balanced_samples = torch.cat(samps, dim=0)
            true_labels_all_samp = true_labels_all[balanced_samples]
            preds_all_samp = preds_all[balanced_samples]
            preds_labels_all_samp = preds_labels_all[balanced_samples]
        else:
            # Use all data points
            true_labels_all_samp = true_labels_all
            preds_all_samp = preds_all
            preds_labels_all_samp = preds_labels_all

        # Log PR curve, ROC curve, confusion matrix, all of the scores from this epoch
        wandb.log(
            {
                f"{phase}_pr_curve": wandb.plot.pr_curve(
                    true_labels_all_samp, preds_all_samp, labels=class_labels,
                ),
                f"{phase}_roc_curve": wandb.plot.roc_curve(
                    true_labels_all_samp, preds_all_samp, labels=class_labels,
                ),
                f"{phase}_conf_mat": wandb.plot.confusion_matrix(
                    y_true=true_labels_all_samp.numpy(),
                    preds=preds_labels_all_samp.numpy(),
                    class_names=class_labels,
                ),
            },
        )
        log_cnt += 1
        print(f"Number of WandB logs for this epoch: {log_cnt}")

    # For progress display
    tqdm.write(f"{phase}_loss: {loss_avg}")

    return loss_avg


# ββββββββββββββββββββββββββββββββββββββββββββββββββββββββββββββββββββββββββββββββ
# ββββββββββββββββββββββββββββββββββββββββββββββββββββββββββββββββββββββββββββββββ
# ββββββββββββββββββββββββββ¦ββββββββββββββββββββββββββββββββββββββββββββββββββββββ
# ββββββββββββββββββββββββββββββββββββββββββββββββββββββββββββββββββββββββββββββββ
# ββββββββββββββββββββββββββ¦ββββββββββββββββββββββββββββββββββββββββββββββββββββββ
# ββββββββββββββββββββββββββββββββββββββββββββββββββββββββββββββββββββββββββββββββ


@typechecked
def save_embeddings(
    data: np.array, loc: str, filename: str, verbose: bool = True
) -> None:
    if verbose:
        print(f"Saving {filename.upper()} embeddings w/ size {data.shape}")
    np.save(f"{loc}{filename}.npy", data)


@typechecked
def create_embeddings(
    config: Mapping, save_emb: bool = False, verbose: bool = True,
) -> None:
    # If you are planning to use BERT as the input to another language model, you would not modify the outputs from BERT.
    # https://www.ncbi.nlm.nih.gov/pmc/articles/PMC8624482/
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
            cat_type=config.cat_type,
            strat_type=config.strat_type,
            max_length=config.max_length,
            sampler=config.sampler,
            split_size=config.split_size,
            complex_graph=config.complex_graph,
            add_more_targets=config.add_more_targets,
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
        )
        if config.saved_model_path:
            logging_to_wandb = False
            from_saved_model = True
            model.load_state_dict(torch.load(config.saved_model_path))
        else:
            logging_to_wandb = False
            from_saved_model = False
        model.to(device)

        # Load dataloaders
        dataloader = get_dataloader(
            data_train=data[f"stratify_{config.strat_type}"]["dataset_train"],
            data_validation=data[f"stratify_{config.strat_type}"]["dataset_valid"],
            data_test=data[f"stratify_{config.strat_type}"]["dataset_test"],
            data_unlabeled=data[f"stratify_{config.strat_type}"]["dataset_unlabeled"],
            class_weights_train=data[f"stratify_{config.strat_type}"][
                "class_weights_train"
            ],
            batch_size=config.batch_size,
            save_emb=save_emb,
        )

        for dataloader, phase in [
            (dataloader["train"], "train"),
            (dataloader["test"], "test"),
            (dataloader["valid"], "val"),
            (dataloader["unlabeled"], "new"),
        ]:
            if verbose:
                print(f"Creating embeddings for {phase.upper()}.")
            # Validation
            valid_epoch(
                model=model,
                dataloader=dataloader,
                epoch=1,
                num_labels=num_labels,
                class_labels=[target2name[lab] for lab in range(num_labels)],
                device=device,
                phase=phase,
                from_saved_model=from_saved_model,
                logging_to_wandb=logging_to_wandb,
                save_emb=save_emb,
            )


@typechecked
def create_bow_embs(
    max_features: int, save_emb: bool = False, verbose: bool = True,
) -> List[pd.DataFrame]:
    embs_df = []
    for complex_graph in [True, False]:
        # Load data
        nodes = load_nodes(
            complex_graph=complex_graph,
            add_more_targets=False,
            frac=1.0,
            seed=SEED,
            verbose=verbose,
        ).fillna("")

        # Pre-process sequence text to remove random NaN strings, digits, and punctuation
        if verbose:
            print("\tPre-process sequence data: remove NAs, digits, punctuation.")
        nodes.loc[:, "sequence"] = (
            nodes.sequence.str.replace("<NA>|NaN", "")
            .str.replace("\d+", "")
            .str.replace(r"[^\w\s]+", "")
        )
        if verbose:
            print("\tApplying CountVectorizer to stripped sequence data.")
        vectorizer = CountVectorizer(max_features=max_features, stop_words="english")
        X = vectorizer.fit_transform(nodes.sequence)
        df_bow_sklearn = pd.DataFrame(
            X.toarray(), columns=vectorizer.get_feature_names()
        )
        embs_df.append(df_bow_sklearn)
        bow_embs = df_bow_sklearn.values

        if save_emb:
            if complex_graph:
                filename = "bow_complex"
            else:
                filename = "bow_simple"
            filename += "_{max_features}"
            save_embeddings(
                data=bow_embs, loc=EMBEDDINGS_PATH, filename=filename, verbose=verbose
            )
    return embs_df


# def train_broad_weighted():
#     """
#     Experiment using
#     β Broad category targets for classification.
#     β Stratification in train/val/test split
#     β NOT normalizing the training class weights for weighted random sampling in the data loader
#     """
#     return train(
#         config=SWEEP_INIT,
#         cat_type="broad",
#         strat_type="sklearn",
#         sampler="norm",
#         save_model=True,
#     )

# def train_ntee_weighted():
#     """
#     Experiment using
#     β NTEE1 category targets for classification.
#     β Stratification in train/val/test split
#     β NOT normalizing the training class weights for weighted random sampling in the data loader
#     """
#     return train(
#         config=SWEEP_INIT,
#         cat_type="ntee",
#         strat_type="sklearn",
#         sampler="norm",
#         save_model=False,
#     )
