# Utility functions
# Common helper functions used across the project.
import logging
import sys
from config import EPOCHS, BATCH_SIZE, LEARNING_RATE, EARLY_STOPPING_PATIENCE

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
