# Configuration settings as an example
import os

# Training hyperparameters
EPOCHS = 1000
EARLY_STOPPING_PATIENCE = 10
BATCH_SIZE = 32
LEARNING_RATE = 0.001

# Paths
DATA_DIR = "./data"
INTERMEDIATE_DIR = "./intermediate"
OUTPUT_DIR = "./output"

BASELINE_MODEL_PATH = os.path.join(OUTPUT_DIR, "baseline_model.joblib")

FASTTEXT_ZIP_PATH = os.path.join(INTERMEDIATE_DIR, "wiki.hu.zip")
FASTTEXT_MODEL_PATH = os.path.join(INTERMEDIATE_DIR, "wiki.hu.bin")

INDIVIDUAL_PATH = os.path.join(INTERMEDIATE_DIR, 'individual_raw.csv')
CONSENSUS_PATH = os.path.join(INTERMEDIATE_DIR, 'consensus_raw.csv')

BASELINE_TRAIN_PATH = os.path.join(INTERMEDIATE_DIR, 'baseline_train.npz')
BASELINE_TEST_PATH = os.path.join(INTERMEDIATE_DIR, 'baseline_test.npz')

FINAL_TRAIN_PATH = os.path.join(INTERMEDIATE_DIR, 'final_train.npz')
FINAL_TEST_PATH = os.path.join(INTERMEDIATE_DIR, 'final_test.npz')

# Urls
DATA_URL = "https://bmeedu-my.sharepoint.com/:u:/g/personal/gyires-toth_balint_vik_bme_hu/IQDYwXUJcB_jQYr0bDfNT5RKARYgfKoH97zho3rxZ46KA1I?e=iFp3iz&download=1&xsdata=MDV8MDJ8fDIyOTc1YmYyMWMzNzQyODFlZWZhMDhkZTM3YmNkMjdifDZhMzU0OGFiNzU3MDQyNzE5MWE4NThkYTAwNjk3MDI5fDB8MHw2MzkwMDk0ODEyNTgwNjE0Njd8VW5rbm93bnxWR1ZoYlhOVFpXTjFjbWwwZVZObGNuWnBZMlY4ZXlKRFFTSTZJbFJsWVcxelgwRlVVRk5sY25acFkyVmZVMUJQVEU5R0lpd2lWaUk2SWpBdU1DNHdNREF3SWl3aVVDSTZJbGRwYmpNeUlpd2lRVTRpT2lKUGRHaGxjaUlzSWxkVUlqb3hNWDA9fDF8TDNSbFlXMXpMekU1T2xSM1NIcHViVlpTVlVKUGFGUjFRVTFyWlc1blIyVlhTSEkzYjB0WVNWQkNTamxxTWtKbkxVdFdkMnN4UUhSb2NtVmhaQzUwWVdOMk1pOWphR0Z1Ym1Wc2N5OHhPVHBVZDBoNmJtMVdVbFZDVDJoVWRVRk5hMlZ1WjBkbFYwaHlOMjlMV0VsUVFrbzVhakpDWnkxTFZuZHJNVUIwYUhKbFlXUXVkR0ZqZGpJdmJXVnpjMkZuWlhNdk1UYzJOVE0xTVRNeU5ETTJPQT09fDBiYmVmZWIwYWJmOTRkZTFlZWZhMDhkZTM3YmNkMjdifGRlNDNhNjEyMWZmNzQxOTk4OGJiYzk4ZWMzZjU4MTdk&sdata=MEVHaDVlSkQrR09NUWRsbFV1SXBTMDNEMDV5OUlWV0hEbmlVcEI5YWNuTT0%3D&ovuser=6a3548ab-7570-4271-91a8-58da00697029%2Cskareerik%40edu.bme.hu"
FASTTEXT_ZIP_URL = "https://dl.fbaipublicfiles.com/fasttext/vectors-wiki/wiki.hu.zip"