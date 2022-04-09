"""All of the methods used with BERT Sequence Classification and hyperparameter tuning w/ W&B"""

# Initialize directory paths, classes, constants, packages, and other methods
from utils import *
from pipeline_helper_methods import *


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
        save_model=False,
    )


def train_broad_weighted():
    """
    Experiment using
    ✓ Broad category targets for classification.
    ✓ Stratification in train/val/test split
    ☓ NOT normalizing the training class weights for weighted random sampling in the data loader
    """
    return train(
        config=SWEEP_INIT,
        cat_type="broad",
        strat_type="sklearn",
        sampler="norm",
        save_model=False,
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
        save_model=False,
    )


def train_ntee_weighted():
    """
    Experiment using
    ✓ NTEE1 category targets for classification.
    ✓ Stratification in train/val/test split
    ☓ NOT normalizing the training class weights for weighted random sampling in the data loader
    """
    return train(
        config=SWEEP_INIT,
        cat_type="ntee",
        strat_type="sklearn",
        sampler="norm",
        save_model=False,
    )


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
                stage="val",
                is_pretrained=True,
                save_emb=False,
            )

        # Save model
        if save_model:
            # Vars
            name = "MODEL"
            d = date.today().strftime("%m-%d")

            # For exporting the model and running inference without defining the model class.
            model_scripted = torch.jit.script(model)  # Export to TorchScript
            model_scripted.save(f"{MODELS_PATH}{name}_{d}")  # Save

            # torch.save(model.state_dict(), f"{MODELS_PATH}{name}_{d}")
        torch.cuda.empty_cache()

        return data


############################
############################
######### TRAINING #########
############################
############################


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


###############################
###############################
######### VALIDATIONN #########
###############################
###############################


@typechecked
def valid_epoch(
    model: BertForSequenceClassification,
    dataloader: DataLoader,
    epoch: int,
    num_labels: int,
    class_labels: Sequence[str],
    device: torch.device,
    stage: str,
    is_pretrained: bool = True,
    save_emb: bool = False,
) -> float:

    # Update state to eval
    model.eval()
    cumu_loss = 0.0

    # For labeling logging variables
    if stage == "test":
        record = "test"
    elif stage == "val":
        record = "val"
    else:
        record = "train"

    # Set up progress bar
    progress_bar = tqdm(
        dataloader, desc=f"Epoch {epoch+1:1d}", leave=False, disable=False,
    )

    # Create empty lists for collecting data to summarize later
    preds_all, true_labels_all, eins_all, embs_all = [], [], np.array([]), []

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
        # (since we won’t be running backprop here)–this just reduces memory consumption and speeds things up a little.
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
        acc = accuracy_weighted(preds, true_labels)
        acc_by_label = accuracy_by_label(preds, true_labels)
        f1 = f1score_weighted(preds, true_labels)
        f1_by_label = f1score_by_label(preds, true_labels)
        # auroc = auroc_weighted(preds, true_labels)

        # Log to W&B and print output
        wandb.log({f"{record}_loss_batch": loss_avg_batch_valid})
        progress_bar.set_postfix(
            {f"{record}_loss_batch": f"{loss_avg_batch_valid:.3f}"}
        )
        preds_all.append(preds)
        true_labels_all.append(true_labels)

    # Calculate average loss for entire validation dataset.
    loss_avg = cumu_loss / len(dataloader)

    # Convert lists to tensors
    true_labels_all = torch.cat(true_labels_all, dim=0)
    preds_all = torch.cat(preds_all, dim=0)
    preds_labels_all = preds_all.argmax(dim=-1)
    embs_all = torch.cat(embs_all, dim=0).numpy()

    # Calculate metrics on all batches using custom accumulation.
    acc = accuracy_weighted.compute()
    acc_by_label = accuracy_by_label.compute()
    f1 = f1score_weighted.compute()
    f1_by_label = f1score_by_label.compute()

    # ROC curve
    # wandb.log({})
    # Confusion matrix
    # wandb.log({})

    # Save embeddings to JSON
    if save_emb:
        if is_pretrained:
            filename = f"pretrained_{record}"
        else:
            filename = record
        save_to_json(
            data=dict(zip(eins, embs_all)), loc=EMBEDDINGS_PATH, filename=record
        )

    # Log to W&B and print output
    epoch_scores = {
        f"{record}_loss": loss_avg,
        f"{record}_acc": acc,
        f"{record}_f1": f1,
    }
    epoch_scores.update(
        {f"{record}_acc_{idx}": val.item() for idx, val in enumerate(acc_by_label)}
    )
    epoch_scores.update(
        {f"{record}_val_f1_{idx}": val.item() for idx, val in enumerate(f1_by_label)}
    )
    wandb.log(epoch_scores)

    # Log PR curve, ROC curve, confusion matrix, all of the scores from this epoch
    wandb.log(
        {
            f"{record}_pr_curve": wandb.plot.pr_curve(
                true_labels_all, preds_all, labels=class_labels,
            ),
            f"{record}_roc_curve": wandb.plot.roc_curve(
                true_labels_all, preds_all, labels=class_labels,
            ),
            f"{record}_conf_mat": wandb.plot.confusion_matrix(
                y_true=true_labels_all.numpy(),
                preds=preds_labels_all.numpy(),
                class_names=class_labels,
            ),
        },
    )
    # For progress display
    tqdm.write(f"{record}_loss: {loss_avg}")

    return loss_avg


######################################
######################################
######### Create Embeddings! #########
######################################
######################################


def create_embeddings(
    config: Mapping,
    cat_type: str = "broad",
    pretrained_model_path: Optional[str] = None,
    is_pretrained: bool = False,
    save_emb: bool = False,
    verbose: bool = True,
):

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

        print("CONFIG")
        print(config)

        # Load dataset
        data = build_data(
            cat_type=cat_type,
            strat_type="none",
            max_length=config.max_length,
            sampler="norm",
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
        if is_pretrained:
            model = torch.jit.load(pretrained_model_path)
        else:
            model = load_model(
                num_labels=num_labels,
                classifier_dropout=config.classifier_dropout,
                verbose=verbose,
            )
        model.to(device)

        # Load dataloaders
        dataloader = get_dataloader(
            data_train=data["stratify_none"]["dataset_train"],
            data_validation=data["stratify_none"]["dataset_valid"],
            data_test=data["stratify_none"]["dataset_test"],
            class_weights_train=data["stratify_none"]["class_weights_train"],
            batch_size=config.batch_size,
        )

        for dataloader, stage in [
            (dataloader["train"], "train"),
            (dataloader["test"], "test"),
            (dataloader["valid"], "val"),
        ]:
            # Validation
            valid_epoch(
                model=model,
                dataloader=dataloader,
                epoch=1,
                num_labels=num_labels,
                class_labels=[target2name[lab] for lab in range(num_labels)],
                device=device,
                stage=stage,
                is_pretrained=is_pretrained,
                save_emb=save_emb,
            )

        return data
