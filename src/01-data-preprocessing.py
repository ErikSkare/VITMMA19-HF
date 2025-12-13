# Data preprocessing script
# This script handles data loading, cleaning, and transformation.
import pandas as pd
from utils import setup_logger, watch
from config import INDIVIDUAL_PATH, CONSENSUS_PATH, TRAIN_PATH, TEST_PATH

logger = setup_logger()

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

def compute_majority(df: pd.DataFrame):
    ratings = df.filter(like="rating_")
    tie_count = ((ratings.values == ratings.max(axis=1).values[:, None]).sum(axis=1) > 1).sum()
    logger.info("Calculating majority ratings (choosing first on tie)")
    logger.info(f"Number of rows with tie in majority votes: {tie_count}")
    df["rating"] = ratings.values.argmax(axis=1) + 1
    return df

# -- PIPELINE -- #
def preprocess():
    logger.info("#####################")
    logger.info("Preprocessing data...")
    logger.info("#####################")

    logger.info(f"Load in: {CONSENSUS_PATH}")
    consensus_df = pd.read_csv(CONSENSUS_PATH)
    logger.info(f"Load in: {INDIVIDUAL_PATH}")
    individual_df = pd.read_csv(INDIVIDUAL_PATH)

    logger.info("> Cleaning consensus dataset")
    consensus_df = clean_consensus(consensus_df)
    consensus_df = compute_majority(consensus_df)

    logger.info("> Cleaning individual dataset")
    individual_df = clean_individual(individual_df)
    individual_df = remove_leakage(individual_df, consensus_df)

    logger.info("> Saving results")
    individual_df.to_csv(TRAIN_PATH, index=False)
    consensus_df.to_csv(TEST_PATH, index=False)
    logger.info(f"Created {TRAIN_PATH}")
    logger.info(f"Created {TEST_PATH}")

if __name__ == "__main__":
    preprocess()
