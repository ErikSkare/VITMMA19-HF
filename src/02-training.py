# Model training script
# This script defines the model architecture and runs the training loop.
import joblib
import torch
import numpy as np
import pandas as pd
import torch.nn as nn
import torchmetrics
import torch.nn.functional as F
from utils import setup_logger, load_config, compute_baseline_features, tokenize_dataset, collate_fn, to_ordinal_levels, to_labels
from config import TRAIN_PATH, BASELINE_MODEL_PATH, CACHED_DATA_PATH, FINAL_MODEL_PATH
from models import build_cnn_model
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split

logger = setup_logger()

seed = 42
np.random.seed(seed)
torch.manual_seed(seed)

# -- BASELINE MODEL -- #
def train_baseline_model(df: pd.DataFrame):
    logger.info("> Training baseline model (logistic regression with L2)")

    logger.info("Extracting hand-crafted features")
    X, y = compute_baseline_features(df["text"]), df["rating"]

    param_grid = {"model__C": np.logspace(-2, 2, 20)}
    model_pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('model', LogisticRegression(penalty="l2"))
    ])

    logger.info("Running hyperparameter optimization")
    logger.info("Searching over: inverse regularization parameter")
    grid = GridSearchCV(model_pipeline, param_grid, cv=10, scoring='neg_mean_absolute_error')
    grid.fit(X, y)

    logger.info(f"Best params: {grid.best_params_}")
    logger.info(f"Best CV score (10 folds): {-grid.best_score_} (MAE estimate)")

    joblib.dump(grid.best_estimator_, BASELINE_MODEL_PATH)
    logger.info(f"Saved baseline model (refitted) to {BASELINE_MODEL_PATH}")

# -- FINAL MODEL -- #
def train_final_model(df: pd.DataFrame, config: dict):
    logger.info("> Training final model (1D CNN)")
    logger.info(f"Using hyperparameters: {config}")

    # Caching
    try:
        data = torch.load(CACHED_DATA_PATH)
        embeddings, targets = data["embeddings"], torch.tensor(data["targets"], dtype=torch.long)
        logger.info("Loaded in tokenized data from cache")
    except: 
        logger.info("Tokenizing dataset")
        embeddings, targets = tokenize_dataset(df["text"]), torch.tensor(df["rating"].values - 1, dtype=torch.long)

        logger.info("Caching tokenized dataset")
        torch.save({'embeddings': embeddings, 'targets': targets}, CACHED_DATA_PATH)

    # Train-val split
    X_train, X_val, y_train, y_val = train_test_split(embeddings, targets, test_size=0.15, stratify=targets, random_state=42)
    logger.info(f"Created train-validation split (train: {len(X_train)}, validation: {len(X_val)})")

    # Create loaders
    train_loader = DataLoader(list(zip(X_train, y_train)), batch_size=config["batch_size"], shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(list(zip(X_val, y_val)), batch_size=config["batch_size"], collate_fn=collate_fn)
    logger.info("Created train and validation DataLoaders")

    # Define model
    model = build_cnn_model(embedding_dim=100)
    logger.info(model)

    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    non_trainable_params = sum(p.numel() for p in model.parameters() if not p.requires_grad)

    logger.info(f"Trainable params: {trainable_params}, non-trainable params: {non_trainable_params}")

    # Optimizer and loss
    optimizer = torch.optim.Adam(model.parameters(), lr=config["lr"])
    scheduler = ReduceLROnPlateau(optimizer)

    # Metrics
    train_loss_metric = torchmetrics.MeanMetric()
    train_mae_metric = torchmetrics.MeanAbsoluteError()

    val_loss_metric = torchmetrics.MeanMetric()
    val_mae_metric = torchmetrics.MeanAbsoluteError()

    best_val_mae = float('inf')
    patience, counter = config["patience"], 0
    torch.save(model.state_dict(), FINAL_MODEL_PATH)

    # Training loop
    for epoch in range(1, config["epochs"] + 1):
        # training
        model.train()
        
        train_loss_metric.reset()
        train_mae_metric.reset()

        for batch_X, batch_y in train_loader:
            optimizer.zero_grad()
            logits = model(batch_X)
            labels = to_labels(logits)
            
            weight = (1 + torch.abs(labels - batch_y)).unsqueeze(1)
            levels = to_ordinal_levels(batch_y)
            loss = F.binary_cross_entropy_with_logits(logits, levels, weight=weight)

            loss.backward()
            optimizer.step()

            train_loss_metric.update(loss)
            train_mae_metric.update(labels, batch_y)

        train_loss = train_loss_metric.compute().item()
        train_mae = train_mae_metric.compute().item()

        # validation
        model.eval()
        
        val_loss_metric.reset()
        val_mae_metric.reset()

        with torch.no_grad():
            for batch_X, batch_y in val_loader:
                logits = model(batch_X)
                labels = to_labels(logits)
                
                weight = (1 + torch.abs(labels - batch_y)).unsqueeze(1)
                levels = to_ordinal_levels(batch_y)
                loss = F.binary_cross_entropy_with_logits(logits, levels, weight=weight)

                val_loss_metric.update(loss)
                val_mae_metric.update(labels, batch_y)

        val_loss = val_loss_metric.compute().item()
        val_mae = val_mae_metric.compute().item()

        # scheduler stepping
        scheduler.step(val_loss)

        # logging
        logger.info(
            f"Epoch {epoch}: "
            f"Train Loss={train_loss:.4f}, Train MAE={train_mae:.4f} | "
            f"Val Loss={val_loss:.4f}, Val MAE={val_mae:.4f} |"
            f"Learning rate={scheduler.get_last_lr()[0]:.1e}"
        )

        # early stopping
        if val_mae < best_val_mae:
            best_val_mae = val_mae
            counter = 0
            torch.save(model.state_dict(), FINAL_MODEL_PATH)
            logger.info(f"Saved best model to {FINAL_MODEL_PATH}")
        else:
            counter += 1
            if counter >= patience:
                logger.info(f"Early stopping")
                return

# -- PIPELINE -- #
def train():
    logger.info("#####################")
    logger.info("Starting training process...")
    logger.info("#####################")

    train_df = pd.read_csv(TRAIN_PATH)
    logger.info(f"> Read in: {TRAIN_PATH}")

    train_baseline_model(train_df)
    train_final_model(train_df, load_config())

if __name__ == "__main__":
    train()
