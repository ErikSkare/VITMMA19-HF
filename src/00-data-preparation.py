# Data preparing script
# This script transforms raw dataset into appropriate format.
import os
import glob
import json
import pandas as pd
import subprocess
from utils import setup_logger
from config import DATA_URL, DATA_DIR, INDIVIDUAL_PATH, CONSENSUS_PATH

logger = setup_logger()

# -- DOWNLOAD, UNZIP -- #
def download_data():
    logger.info("> Downloading raw dataset")
    output_path = os.path.join(DATA_DIR, 'raw.zip')
    subprocess.run(["wget", "-q", DATA_URL, "-O", output_path])

def unzip_data():
    logger.info("> Unzipping raw dataset")
    output_path = os.path.join(DATA_DIR, 'raw.zip')
    subprocess.run(["unzip", "-q", "-o", output_path, "-d", DATA_DIR])

# -- PREPARE -- #
def safe_get(entry, path, dtype):
    try: 
        current = entry
        for node in path: current = current[node]
        return dtype(current)
    except (KeyError, IndexError, TypeError): 
        return pd.NA

def process_file(path: str) -> pd.DataFrame:
    with open(path, 'r') as file: data = json.load(file)
    return pd.DataFrame([
        {
            'text':   safe_get(entry, ['data', 'text'], str),
            'rating': safe_get(entry, ['annotations', 0, 'result', 0, 'value', 'choices', 0, 0], int)
        }
        for entry in data
    ])

def merge_disjunct(*dfs) -> pd.DataFrame:
    merged = pd.concat(dfs, ignore_index=True)
    merged.reset_index(drop=True, inplace=True)  
    return merged

def merge_paired(*dfs) -> pd.DataFrame:
    df = pd.concat(dfs, ignore_index=True)
    df = pd.get_dummies(df, columns=['rating'])
    return df.groupby('text', as_index=False).sum().sort_values(by='text')

# -- PIPELINE -- #
def prepare():
    logger.info("#####################")
    logger.info('Preparing dataset...')
    logger.info("#####################")

    download_data()
    unzip_data()

    all_paths = glob.glob(os.path.join(DATA_DIR, 'legaltextdecoder', '**', '*.json'))
    consensus_paths = glob.glob(os.path.join(DATA_DIR, 'legaltextdecoder', 'consensus', '*.json'), )
    individual_paths = [p for p in all_paths if p not in consensus_paths]

    logger.info(f"Found: {len(individual_paths)} individual files, {len(consensus_paths)} consensus files.")

    if len(all_paths) == 0: return

    individual_df = merge_disjunct(*[process_file(path) for path in individual_paths])
    consensus_df = merge_paired(*[process_file(path) for path in consensus_paths])

    logger.info(f"Read in: {len(individual_df)} individual paragraphs, {len(consensus_df)} consensus paragraphs.")

    individual_df.to_csv(INDIVIDUAL_PATH, index=False)
    consensus_df.to_csv(CONSENSUS_PATH, index=False)

    logger.info(f'Created {INDIVIDUAL_PATH}')
    logger.info(f'Created {CONSENSUS_PATH}')

if __name__ == '__main__':
    prepare()
