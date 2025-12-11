# Utility functions
# Common helper functions used across the project.
import logging
import sys
import os
import subprocess
import fasttext
from config import INTERMEDIATE_DIR, FASTTEXT_ZIP_URL, FASTTEXT_ZIP_PATH, FASTTEXT_MODEL_PATH

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

def get_fasttext_model():
    if not os.path.exists(FASTTEXT_ZIP_PATH):
        subprocess.run(["wget", "-q", "--show-progress", FASTTEXT_ZIP_URL, "-O", FASTTEXT_ZIP_PATH])
        subprocess.run(["unzip", "-q", "-o", FASTTEXT_ZIP_PATH, "-d", INTERMEDIATE_DIR])
    return fasttext.load_model(FASTTEXT_MODEL_PATH)
