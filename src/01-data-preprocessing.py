# Data preprocessing script
# This script handles data loading, cleaning, and transformation.
import pandas as pd
from utils import setup_logger
from config import INDIVIDUAL_RAW_PATH, CONSENSUS_RAW_PATH, TRAIN_DATA_PATH, TEST_DATA_PATH

logger = setup_logger()

def watch(df, fn):
    before = len(df)
    df = fn(df)
    after = len(df)
    return df, before - after

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
    df["majority_rating"] = ratings.values.argmax(axis=1) + 1
    return df

def preprocess():
    logger.info("Preprocessing data...")

    logger.info(f"Load in: {CONSENSUS_RAW_PATH}")
    consensus_df = pd.read_csv(CONSENSUS_RAW_PATH)
    logger.info(f"Load in: {INDIVIDUAL_RAW_PATH}")
    individual_df = pd.read_csv(INDIVIDUAL_RAW_PATH)

    logger.info("Processing consensus dataset...")
    consensus_df = clean_consensus(consensus_df)
    consensus_df = append_majority(consensus_df)

    logger.info("Processing individual dataset...")
    individual_df = clean_individual(individual_df)
    individual_df = remove_leakage(individual_df, consensus_df)

    individual_df.to_csv(TRAIN_DATA_PATH, index=False)
    consensus_df.to_csv(TEST_DATA_PATH, index=False)
    logger.info(f"Created {TRAIN_DATA_PATH} from processed individual dataset")
    logger.info(f"Created {TEST_DATA_PATH} from processed consensus dataset")

if __name__ == "__main__":
    preprocess()
