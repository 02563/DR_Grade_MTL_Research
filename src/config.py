### config.py
import os

class Config:
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    BASE_DIR2 = '/openbayes/input/input0'

    RAW_DIR = BASE_DIR2
    PROCESSED_DIR = os.path.join(BASE_DIR, "data", "processed")

    CSV_PATH = os.path.join(BASE_DIR2, "trainLabels.csv")

    IMG_PARAMS = {
        "INPUT_SIZE": 224,
        "CROP_SIZE": 256
    }

    TRAIN_PARAMS = {
        "EPOCHS": 60,
        "BATCH_SIZE": 64,
        "LR": 5e-5,
        "WARMUP_EPOCHS": 2,
        "VAL_SPLIT": 0.2,
        "NUM_TRAIN_SAMPLES": 11812,
        "NUM_VAL_SAMPLES": 2954,
        "UNFREEZE_EPOCH": 5,
        "GRADIENT_CLIP": 1.0
    }

    MODEL_PARAMS = {
        "NUM_CLASSES": 5,
        "DROPOUT": 0.5,
        "WEIGHT_DECAY": 1e-4
    }

    CHECKPOINT_PATH = os.path.join(BASE_DIR, "checkpoints", "best_model")
    LOG_DIR = os.path.join(BASE_DIR, "logs")

config = Config()