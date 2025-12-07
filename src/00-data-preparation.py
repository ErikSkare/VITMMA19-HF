import glob
import json
import pandas as pd
from utils import setup_logger
from config import INDIVIDUAL_DATA_PATH, CONSENSUS_DATA_PATH

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

def prepare():
    logger.info('Preparing dataset...')

    all_paths = glob.glob('./data/**/*.json')
    consensus_paths = glob.glob('./data/consensus/*.json')
    individual_paths = [p for p in all_paths if p not in consensus_paths]

    logger.info(f"Found: {len(individual_paths)} individual files, {len(consensus_paths)} consensus files.")

    individual_df = merge_disjunct(*[process_file(path) for path in individual_paths])
    consensus_df = merge_paired(*[process_file(path) for path in consensus_paths])

    logger.info(f"Read in: {len(individual_df)} individual paragraphs, {len(consensus_df)} consensus paragraphs.")

    individual_df.to_csv(INDIVIDUAL_DATA_PATH, index=False)
    consensus_df.to_csv(CONSENSUS_DATA_PATH, index=False)

    logger.info(f'Created {INDIVIDUAL_DATA_PATH}.')
    logger.info(f'Created {CONSENSUS_DATA_PATH}.')

    logger.info('Preparation is finished...')

if __name__ == '__main__':
    prepare()
