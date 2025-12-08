# Data preprocessing script
# This script handles data loading, cleaning, and transformation.
import pandas as pd
from utils import setup_logger
from config import INDIVIDUAL_DATA_PATH, CONSENSUS_DATA_PATH, INDIVIDUAL_CLEANED_PATH, CONSENSUS_CLEANED_PATH

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
    logger.info(f"Deleted {affected} leaking rows from individual dataset")
    return df

def preprocess():
    logger.info("Preprocessing data...")

    logger.info(f"Load in: {INDIVIDUAL_DATA_PATH}")
    individual_df = pd.read_csv(INDIVIDUAL_DATA_PATH)
    logger.info(f"Load in: {CONSENSUS_DATA_PATH}")
    consensus_df = pd.read_csv(CONSENSUS_DATA_PATH)

    logger.info("Cleaning individual dataset...")
    individual_df = clean_individual(individual_df)

    logger.info("Cleaning consensus dataset...")
    consensus_df = clean_consensus(consensus_df)

    individual_df = remove_leakage(individual_df, consensus_df)

    individual_df.to_csv(INDIVIDUAL_CLEANED_PATH, index=False)
    consensus_df.to_csv(CONSENSUS_CLEANED_PATH, index=False)
    logger.info(f"Created {INDIVIDUAL_CLEANED_PATH}")
    logger.info(f"Created {CONSENSUS_CLEANED_PATH}")

if __name__ == "__main__":
    preprocess()
