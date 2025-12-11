# Data preprocessing script
# This script handles data loading, cleaning, and transformation.
import numpy as np
import pandas as pd
import string
import re
from utils import setup_logger, get_fasttext_model
from config import INDIVIDUAL_PATH, CONSENSUS_PATH, BASELINE_TRAIN_PATH, BASELINE_TEST_PATH, FINAL_TRAIN_PATH, FINAL_TEST_PATH

logger = setup_logger()

def watch(df, fn):
    before = len(df)
    df = fn(df)
    after = len(df)
    return df, before - after

# -- DATA CLEANING -- #
def clean_individual(df: pd.DataFrame) -> pd.DataFrame:
    df, affected = watch(df, lambda df: df.dropna().reset_index(drop=True))
    logger.info(f"Deleted {affected} NA rows")
    df, affected = watch(df, lambda df: df.drop_duplicates(['text'], keep='first').reset_index(drop=True))
    logger.info(f"Deleted {affected} duplicated rows (keep first)")
    return df

def clean_consensus(df: pd.DataFrame) -> pd.DataFrame:
    sums = df.filter(like="rating_").sum(axis=1)
    df, affected = watch(df, lambda df: df[sums >= 2].reset_index(drop=True))
    logger.info(f"Deleted {affected} individual rows")
    return df

def remove_leakage(df: pd.DataFrame, ref_df: pd.DataFrame) -> pd.DataFrame:
    df, affected = watch(df, lambda df: df[~df['text'].isin(ref_df['text'])].reset_index(drop=True))
    logger.info(f"Deleted {affected} leaking rows")
    return df

def append_majority(df: pd.DataFrame):
    ratings = df.filter(like="rating_")
    tie_count = ((ratings.values == ratings.max(axis=1).values[:, None]).sum(axis=1) > 1).sum()
    logger.info("Calculating majority ratings (choosing first on tie)")
    logger.info(f"Number of rows with tie in majority votes: {tie_count}")
    df["rating"] = ratings.values.argmax(axis=1) + 1
    return df

# -- DATA TRANSFORMATION -- #
def compute_baseline_features(df: pd.DataFrame):
    features = pd.DataFrame()
    features["char_count"] = df["text"].apply(lambda x: len(str(x)))
    features["word_count"] = df["text"].apply(lambda x: len(str(x).split()))
    features["sentence_count"] = df["text"].apply(lambda x: len(re.findall(r'[.!?]+', x)))
    features["avg_word_length"] = df["text"].apply(lambda x: sum(len(word) for word in str(x).split()) / max(len(str(x).split()), 1))
    features["punctuation_count"] = df["text"].apply(lambda x: sum(1 for c in str(x) if c in string.punctuation))
    return (features.to_numpy(), df["rating"].to_numpy(dtype=np.int32))

def embed_sentence(sentence: str, ft_model, chunk_size: int = 256):
    words = sentence.split()
    chunks = [words[i:i+chunk_size] for i in range(0, len(words), chunk_size)]

    embeddings = []
    for chunk in chunks:
        matrix = np.zeros((chunk_size, 300), dtype=np.float32)
        for i, word in enumerate(chunk):
            matrix[i] = ft_model.get_word_vector(word)
        embeddings.append(matrix)
    return embeddings

def compute_fasttext_embeddings(df: pd.DataFrame, ft_model):
    doc_ids, features, labels = [], [], []
    for id, row in df.iterrows():
        embeddings = embed_sentence(row["text"], ft_model)
        features.extend(embeddings)
        doc_ids.extend([id for _ in range(len(embeddings))])
        labels.extend([row["rating"] for _ in range(len(embeddings))])
    return (np.array(doc_ids, dtype=np.int32), np.array(features), np.array(labels, dtype=np.int32))

def preprocess():
    logger.info("#####################")
    logger.info("Preprocessing data...")

    logger.info(f"Load in: {CONSENSUS_PATH}")
    consensus_df = pd.read_csv(CONSENSUS_PATH)
    logger.info(f"Load in: {INDIVIDUAL_PATH}")
    individual_df = pd.read_csv(INDIVIDUAL_PATH)

    logger.info("Cleaning consensus dataset...")
    consensus_df = clean_consensus(consensus_df)
    consensus_df = append_majority(consensus_df)

    logger.info("Cleaning individual dataset...")
    individual_df = clean_individual(individual_df)
    individual_df = remove_leakage(individual_df, consensus_df)

    logger.info("Computing baseline features...")
    baseline_train = compute_baseline_features(individual_df)
    np.savez(BASELINE_TRAIN_PATH, features=baseline_train[0], labels=baseline_train[1])
    logger.info(f"Created {BASELINE_TRAIN_PATH}")

    baseline_test = compute_baseline_features(consensus_df)
    np.savez(BASELINE_TEST_PATH, features=baseline_test[0], labels=baseline_test[1])
    logger.info(f"Created {BASELINE_TEST_PATH}")

    logger.info("Computing FastText embeddings for model training (chunk size = 256)...")
    ft_model = get_fasttext_model()

    final_train = compute_fasttext_embeddings(individual_df, ft_model)
    np.savez(FINAL_TRAIN_PATH, doc_ids=final_train[0], features=final_train[1], labels=final_train[2])
    logger.info(f"Created {FINAL_TRAIN_PATH}")

    final_test = compute_fasttext_embeddings(consensus_df, ft_model)
    np.savez(FINAL_TEST_PATH, doc_ids=final_test[0], features=final_test[1], labels=final_test[2])
    logger.info(f"Created {FINAL_TEST_PATH}")
    logger.info("#####################")

if __name__ == "__main__":
    preprocess()