# Data preparing script
# This script transforms raw dataset into appropriate format.
import argparse
import os
import glob
import json
import pandas as pd
from utils import setup_logger
from config import INDIVIDUAL_RAW_PATH, CONSENSUS_RAW_PATH

logger = setup_logger()

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

def prepare(data_folder: str):
    logger.info('Preparing dataset...')
    logger.info(f'Data folder: {data_folder}')

    all_paths = glob.glob(os.path.join(data_folder, '**', '*.json'))
    consensus_paths = glob.glob(os.path.join(data_folder, 'consensus', '*.json'), )
    individual_paths = [p for p in all_paths if p not in consensus_paths]

    logger.info(f"Found: {len(individual_paths)} individual files, {len(consensus_paths)} consensus files.")

    if len(all_paths) == 0: return

    individual_df = merge_disjunct(*[process_file(path) for path in individual_paths])
    consensus_df = merge_paired(*[process_file(path) for path in consensus_paths])

    logger.info(f"Read in: {len(individual_df)} individual paragraphs, {len(consensus_df)} consensus paragraphs.")

    individual_df.to_csv(INDIVIDUAL_RAW_PATH, index=False)
    consensus_df.to_csv(CONSENSUS_RAW_PATH, index=False)

    logger.info(f'Created {INDIVIDUAL_RAW_PATH}.')
    logger.info(f'Created {CONSENSUS_RAW_PATH}.')

    logger.info('Preparation is finished...')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Prepare dataset")
    parser.add_argument(
        '--data-folder', 
        type=str, 
        default='./data',
        help='Path to the folder containing raw data (defaults to ./data)'
    )
    args = parser.parse_args()

    prepare(args.data_folder)
