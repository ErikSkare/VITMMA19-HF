# Model training script
# This script defines the model architecture and runs the training loop.
import re
import string
import joblib
import torch
import huspacy
import numpy as np
import pandas as pd
import torch.nn as nn
import torchmetrics
from utils import setup_logger
from config import TRAIN_PATH, BASELINE_MODEL_PATH, CACHED_DATA_PATH
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from torch.nn.utils.rnn import pad_sequence

logger = setup_logger()
tokenizer = huspacy.load("hu_core_news_md", disable=["tagger","parser","ner"])

# -- BASELINE MODEL -- #
def compute_baseline_features(texts: pd.Series):
    features = pd.DataFrame()
    features["char_count"] = texts.apply(lambda x: len(str(x)))
    features["word_count"] = texts.apply(lambda x: len(str(x).split()))
    features["sentence_count"] = texts.apply(lambda x: len(re.findall(r'[.!?]+', x)))
    features["avg_word_length"] = texts.apply(lambda x: sum(len(word) for word in str(x).split()) / max(len(str(x).split()), 1))
    features["punctuation_count"] = texts.apply(lambda x: sum(1 for c in str(x) if c in string.punctuation))
    return features

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
    grid = GridSearchCV(model_pipeline, param_grid, cv=10, scoring='balanced_accuracy')
    grid.fit(X, y)

    logger.info(f"Best params: {grid.best_params_}")
    logger.info(f"Best CV score (10 folds): {grid.best_score_} (balanced accuracy estimate)")

    joblib.dump(grid.best_estimator_, BASELINE_MODEL_PATH)
    logger.info(f"Saved baseline model (refitted) to {BASELINE_MODEL_PATH}")

# -- FINAL MODEL -- #
def tokenize_paragraph(text: str):
    doc = tokenizer(text)
    embeddings = np.array([token.vector for token in doc])
    return torch.tensor(embeddings)

def tokenize_dataset(texts: pd.Series):
    embeddings, lengths = [], []
    for text in texts: 
        current = tokenize_paragraph(text)
        embeddings.append(current)
        lengths.append(len(current))
    
    lengths = torch.tensor(lengths)
    embeddings = pad_sequence(embeddings, batch_first=True)

    mask = (torch.arange(lengths.max()) < lengths.unsqueeze(1)).int()
    mask = mask.unsqueeze(-1).expand(-1, -1, embeddings.size(-1))
    
    combined = torch.stack([embeddings, mask], dim=1)
    return combined

class FinalModel(nn.Module):
    def __init__(self, embedding_dim: int, num_classes: int = 5):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(in_channels=embedding_dim, out_channels=16, kernel_size=3),
            nn.BatchNorm1d(16),
            nn.ReLU(),
        )
        self.hidden = nn.Sequential(
            nn.AdaptiveMaxPool1d(1),
            nn.Flatten(),
        )
        self.linear = nn.Sequential(
            nn.Linear(16, 8),
            nn.Linear(8, num_classes)
        )
    
    def forward(self, x):
        # Extracting mask and embeddings
        mask = x[:, 1]
        x = x[:, 0]
        # Convolution
        x = x.transpose(1, 2)
        x = self.conv(x)
        # Masking
        mask = mask.transpose(1, 2)
        x = x.masked_fill(mask == 0, -1e9)
        # Pooling
        x = self.hidden(x)
        # Classification
        logits = self.linear(x)
        return logits

def train_final_model(df: pd.DataFrame):
    logger.info("> Training final model (1D CNN)")

    # Caching
    try:
        data = torch.load(CACHED_DATA_PATH)
        X_train, X_val = data["X_train"], data["X_val"]
        y_train, y_val = data["y_train"], data["y_val"]
        logger.info("Loaded in tokenized data from cache")
    except: 
        # Train-val split
        train_df, val_df = train_test_split(df, test_size=0.05, stratify=df["rating"], random_state=42)
        logger.info(f"Splitting dataset (train: {len(train_df)}, val: {len(val_df)})")

        # Embeddings
        logger.info("Tokenizing splitted datasets")
        X_train, y_train = tokenize_dataset(train_df["text"]), torch.tensor(train_df["rating"].values, dtype=torch.long)
        X_val, y_val = tokenize_dataset(val_df["text"]), torch.tensor(val_df["rating"].values, dtype=torch.long)

        # Caching
        torch.save({'X_train': X_train, 'y_train': y_train, 'X_val': X_val, 'y_val': y_val}, CACHED_DATA_PATH)
        logger.info("Caching tokenized datasets")

    # Normalize
    logger.info("Normalizing embedding vectors")
    X_train = X_train / (np.linalg.norm(X_train, axis=-1, keepdims=True) + 1e-8)
    X_val = X_val / (np.linalg.norm(X_val, axis=-1, keepdims=True) + 1e-8)

    # Permute to (batch, channels, sequence_length)
    X_train, X_val = X_train.permute(0, 2, 1), X_val.permute(0, 2, 1)

    # Create loaders
    train_dataset, val_dataset = TensorDataset(X_train, y_train), TensorDataset(X_val, y_val)
    train_loader, val_loader = DataLoader(train_dataset, batch_size=256, shuffle=True), DataLoader(val_dataset, batch_size=256)

    # Define model
    model = FinalModel(embedding_dim=100)

    # Optimizer and loss
    class_counts = torch.bincount(y_train)
    weight = 1.0 / class_counts.float()
    weight /= weight.sum()

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss(weight=weight)

    # Metrics
    train_loss_metric = torchmetrics.MeanMetric()
    train_acc_metric = torchmetrics.Accuracy(task="multiclass", num_classes=5, average="macro")

    val_loss_metric = torchmetrics.MeanMetric()
    val_acc_metric = torchmetrics.Accuracy(task="multiclass", num_classes=5, average="macro")

    # Training loop
    for epoch in range(1, 1001):
        # training
        model.train()
        
        train_loss_metric.reset()
        train_acc_metric.reset()

        for batch_X, batch_y in train_loader:
            optimizer.zero_grad()
            logits = model(batch_X)
            
            loss = criterion(logits, batch_y)
            loss.backward()
            optimizer.step()

            train_loss_metric.update(loss)
            train_acc_metric.update(logits, batch_y)

        train_loss = train_loss_metric.compute().item()
        train_acc = train_acc_metric.compute().item()

        # validation
        model.eval()
        
        val_loss_metric.reset()
        val_acc_metric.reset()

        with torch.no_grad():
            for batch_X, batch_y in val_loader:
                logits = model(batch_X)
                loss = criterion(logits, batch_y)

                val_loss_metric.update(loss)
                val_acc_metric.update(logits, batch_y)

        val_loss = val_loss_metric.compute().item()
        val_acc = val_acc_metric.compute().item()

        print(
            f"Epoch {epoch}: "
            f"Train Loss={train_loss:.4f}, Train Acc={train_acc:.4f} | "
            f"Val Loss={val_loss:.4f}, Val Acc={val_acc:.4f}"
        )

# -- PIPELINE -- #
def train():
    logger.info("#####################")
    logger.info("Starting training process...")
    logger.info("#####################")

    train_df = pd.read_csv(TRAIN_PATH)
    logger.info(f"> Read in: {TRAIN_PATH}")

    train_final_model(train_df)


if __name__ == "__main__":
    train()
