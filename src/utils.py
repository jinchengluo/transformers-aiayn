import os
from pathlib import Path

# Transformer model parameters
MODEL_NUMBER_OF_LAYERS = 3
MODEL_DIMENSION = 256
MODEL_NUMBER_OF_HEADS = 4
MODEL_INNER_LAYER_DIMENSION = 1024
MODEL_DROPOUT_PROBABILITY = 0.1
MODEL_LABEL_SMOOTHING_VALUE = 0.1
SEQUENCE_LENGTH = 258

# Training parameters
BATCH_SIZE = 8
NUMBER_OF_EPOCHS = 10
BETA1 = 0.9
BETA2 = 0.98
EPSILON = 1e-9
WARMUP_STEPS = 4000

# Dataset parameters
DATASET_NAME = "Helsinki-NLP/opus_books"
SOURCE_LANGUAGE = "en"
TARGET_LANGUAGE = "fr"

# Saving parameters
MODEL_FOLDER = "weights"
MODEL_BASENAME = "tmodel_"
MODEL_PRELOAD = "latest"
EXPERIMENT_FOLDER = "runs/tmodel"

# Special tokens
UNK_TOKEN = '[UNK]'
SOS_TOKEN = '[SOS]'
EOS_TOKEN = '[EOS]'
PAD_TOKEN = "[PAD]"

CHECKPOINTS_PATH = os.path.join(os.getcwd(), 'models', 'checkpoints') # semi-trained models during training will be dumped here
BINARIES_PATH = os.path.join(os.getcwd(), 'models', 'binaries') # location where trained models are located
DATA_DIR_PATH = os.path.join(os.getcwd(), 'data') # training data will be stored here

os.makedirs(CHECKPOINTS_PATH, exist_ok=True)
os.makedirs(BINARIES_PATH, exist_ok=True)
os.makedirs(DATA_DIR_PATH, exist_ok=True)

def get_weights_file_path(epoch: str):
    model_folder = f"{DATASET_NAME}_{MODEL_FOLDER}"
    model_filename = f"{MODEL_BASENAME}{epoch}.pt"
    return str(Path('.') / model_folder / model_filename)

# Find the latest weights file in the weights folder
def latest_weights_file_path():
    model_folder = f"{DATASET_NAME}_{MODEL_FOLDER}"
    model_filename = f"{MODEL_BASENAME}*"
    weights_files = list(Path(model_folder).glob(model_filename))
    if len(weights_files) == 0:
        return None
    weights_files.sort()
    return str(weights_files[-1])