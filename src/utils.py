# Utility functions
# Common helper functions used across the project.
import logging
import sys
import pandas as pd
import string
import re
import importlib
import huspacy
import torch
import subprocess
import numpy as np
from config import EPOCHS, BATCH_SIZE, LEARNING_RATE, EARLY_STOPPING_PATIENCE, HUSPACY_MODEL_URL
from torch.nn.utils.rnn import pad_sequence

# General
def setup_logger(name=__name__):
    """
    Sets up a logger that outputs to the console (stdout).
    """
    logger = logging.getLogger(name)
    if not logger.handlers:
        logger.setLevel(logging.INFO)
        handler = logging.StreamHandler(sys.stdout)
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    return logger

def watch(df, fn):
    before = len(df)
    df = fn(df)
    after = len(df)
    return df, before - after

def load_config():
    return {
        "epochs": EPOCHS,
        "batch_size": BATCH_SIZE,
        "lr": LEARNING_RATE,
        "patience": EARLY_STOPPING_PATIENCE,
    }

# For baseline model
logger = setup_logger()

model_name = "hu_core_news_md"
if importlib.util.find_spec(model_name) is None:
    logger.info(f"Downloading huSpaCy model '{model_name}'...")
    subprocess.run(
        ["pip", "install", HUSPACY_MODEL_URL],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        check=True
    )
else:
    logger.info(f"Model '{model_name}' already installed.")

tokenizer = huspacy.load("hu_core_news_md", disable=["tagger","parser","ner"])

def compute_baseline_features(texts: pd.Series):
    features = pd.DataFrame()
    features["char_count"] = texts.apply(lambda x: len(str(x)))
    features["word_count"] = texts.apply(lambda x: len(str(x).split()))
    features["sentence_count"] = texts.apply(lambda x: len(re.findall(r'[.!?]+', x)))
    features["avg_word_length"] = texts.apply(lambda x: sum(len(word) for word in str(x).split()) / max(len(str(x).split()), 1))
    features["punctuation_count"] = texts.apply(lambda x: sum(1 for c in str(x) if c in string.punctuation))
    return features

# For final model
def tokenize_paragraph(text: str):
    doc = tokenizer(text)
    embeddings = np.array([token.vector for token in doc])
    embeddings /= (np.linalg.norm(embeddings, axis=-1, keepdims=True) + 1e-8)
    return torch.tensor(embeddings)

def tokenize_dataset(texts: pd.Series):
    embeddings = []
    for text in texts: embeddings.append(tokenize_paragraph(text))
    return embeddings

def collate_fn(batch):
    embeddings, targets = zip(*batch)
    embeddings = pad_sequence(embeddings, batch_first=True)
    embeddings = embeddings.permute(0, 2, 1)
    return embeddings, torch.tensor(targets, dtype=torch.long)

def to_ordinal_levels(batch):
    ordinal_targets = torch.zeros((batch.size(0), 4))
    for k in range(4): ordinal_targets[:, k] = (batch > k).float()
    return ordinal_targets

def to_labels(logits):
    probs = torch.sigmoid(logits)
    binary = (probs > 0.5).int()
    return binary.cumprod(dim=1).sum(dim=1)
