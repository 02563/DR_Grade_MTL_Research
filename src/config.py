import os

class Config:
    """全局配置"""
    # 项目根路径
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    # 数据路径
    RAW_DIR = os.path.join(BASE_DIR, "data", "raw")
    PROCESSED_DIR = os.path.join(BASE_DIR, "data", "processed")

    # CSV文件路径
    CSV_PATH = os.path.join(PROCESSED_DIR, "combined_train.csv")

    # 图片处理参数
    IMG_PARAMS = {
        "INPUT_SIZE": 224,
        "CROP_SIZE": 256
    }

    # 训练参数
    TRAIN_PARAMS = {
        "EPOCHS": 40,
        "BATCH_SIZE": 32,
        "LR": 1e-3,
        "WARMUP_EPOCHS": 1,
        "VAL_SPLIT": 0.2,
        "NUM_TRAIN_SAMPLES": 9656,  # 后续在create_tfrecords动态更新
        "NUM_VAL_SAMPLES": 2414,
        "UNFREEZE_EPOCH": 3,
        "GRADIENT_CLIP": 5.0
    }

    MODEL_PARAMS = {
        "NUM_CLASSES": 5   # APTOS竞赛是5类
    }

    # 其他路径
    CHECKPOINT_PATH = os.path.join(BASE_DIR, "checkpoints", "best_model.h5")
    LOG_DIR = os.path.join(BASE_DIR, "logs")

config = Config()
