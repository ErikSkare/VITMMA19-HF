# Configuration settings as an example
import os

# Training hyperparameters
EPOCHS = 1000
BATCH_SIZE = 128
LEARNING_RATE = 1e-4
EARLY_STOPPING_PATIENCE = 20

# Paths
DATA_DIR = "/app/data"
INTERMEDIATE_DIR = "/app/intermediate"
OUTPUT_DIR = "/app/output"

BASELINE_MODEL_PATH = os.path.join(OUTPUT_DIR, "baseline_model.joblib")
FINAL_MODEL_PATH = os.path.join(OUTPUT_DIR, "final_model.pth")

INDIVIDUAL_PATH = os.path.join(INTERMEDIATE_DIR, 'individual_raw.csv')
CONSENSUS_PATH = os.path.join(INTERMEDIATE_DIR, 'consensus_raw.csv')
TRAIN_PATH = os.path.join(INTERMEDIATE_DIR, 'train_data.csv')
TEST_PATH = os.path.join(INTERMEDIATE_DIR, 'test_data.csv')
CACHED_DATA_PATH = os.path.join(INTERMEDIATE_DIR, 'cached.pt')

# Urls
DATA_URL = "https://bmeedu-my.sharepoint.com/:u:/g/personal/gyires-toth_balint_vik_bme_hu/IQDYwXUJcB_jQYr0bDfNT5RKARYgfKoH97zho3rxZ46KA1I?e=iFp3iz&download=1&xsdata=MDV8MDJ8fDIyOTc1YmYyMWMzNzQyODFlZWZhMDhkZTM3YmNkMjdifDZhMzU0OGFiNzU3MDQyNzE5MWE4NThkYTAwNjk3MDI5fDB8MHw2MzkwMDk0ODEyNTgwNjE0Njd8VW5rbm93bnxWR1ZoYlhOVFpXTjFjbWwwZVZObGNuWnBZMlY4ZXlKRFFTSTZJbFJsWVcxelgwRlVVRk5sY25acFkyVmZVMUJQVEU5R0lpd2lWaUk2SWpBdU1DNHdNREF3SWl3aVVDSTZJbGRwYmpNeUlpd2lRVTRpT2lKUGRHaGxjaUlzSWxkVUlqb3hNWDA9fDF8TDNSbFlXMXpMekU1T2xSM1NIcHViVlpTVlVKUGFGUjFRVTFyWlc1blIyVlhTSEkzYjB0WVNWQkNTamxxTWtKbkxVdFdkMnN4UUhSb2NtVmhaQzUwWVdOMk1pOWphR0Z1Ym1Wc2N5OHhPVHBVZDBoNmJtMVdVbFZDVDJoVWRVRk5hMlZ1WjBkbFYwaHlOMjlMV0VsUVFrbzVhakpDWnkxTFZuZHJNVUIwYUhKbFlXUXVkR0ZqZGpJdmJXVnpjMkZuWlhNdk1UYzJOVE0xTVRNeU5ETTJPQT09fDBiYmVmZWIwYWJmOTRkZTFlZWZhMDhkZTM3YmNkMjdifGRlNDNhNjEyMWZmNzQxOTk4OGJiYzk4ZWMzZjU4MTdk&sdata=MEVHaDVlSkQrR09NUWRsbFV1SXBTMDNEMDV5OUlWV0hEbmlVcEI5YWNuTT0%3D&ovuser=6a3548ab-7570-4271-91a8-58da00697029%2Cskareerik%40edu.bme.hu"
HUSPACY_MODEL_URL="https://huggingface.co/huspacy/hu_core_news_md/resolve/main/hu_core_news_md-any-py3-none-any.whl"