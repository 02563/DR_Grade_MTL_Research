import os

class Config:
    def __init__(self):
        # 数据集路径
        self.DATA_ROOT = r"E:\DR_Grade_MTL_Research\data"
        self.RAW_DIR = os.path.join(self.DATA_ROOT, "raw/aptos2019-blindness-detection")
        self.PROCESSED_DIR = os.path.join(self.DATA_ROOT, "processed")  # 确保此目录生成

        #self.DATA_ROOT = "E:/DR_Grade_MTL_Research/data"
        #self.RAW_DIR = os.path.join(self.DATA_ROOT, "raw/aptos2019-blindness-detection")
        #self.PROCESSED_DIR = os.path.join(self.DATA_ROOT, "processed")
        self.CSV_PATH = os.path.join(self.RAW_DIR, "train.csv")
        
        # 图像参数
        self.IMG_PARAMS = {
            "IMG_SIZE": 384,
            "CROP_SIZE": 224,
            "INPUT_SIZE": 224  
        }

        # 训练参数
        self.TRAIN_PARAMS = {
            "BATCH_SIZE": 4,  # 批量大小
            "EPOCHS": 40,
            "LR": 1e-4,
            "VAL_SPLIT": 0.2,
            "USE_FOCAL_LOSS": True,
            "GRADIENT_CLIP": 1.0,
            "WARMUP_EPOCHS": 5,
            "NUM_TRAIN_SAMPLES": 2929,  # 初始化后自动填充
            "NUM_VAL_SAMPLES": 733
        }

        # 多任务参数
        self.TASKS = {
            'grade': {
                'type': 'classification',
                'num_classes': 5,
                'loss_weight': 1.0
            },
            'recon': {
                'type': 'regression'
            }
        }

        # 路径配置
        self.CHECKPOINT_PATH = "./checkpoints/best_model.h5"
        self.LOG_DIR = "./logs"
        self._ensure_directories()

    def _ensure_directories(self):
        """创建必要目录"""
        os.makedirs(self.PROCESSED_DIR, exist_ok=True)
        os.makedirs(os.path.dirname(self.CHECKPOINT_PATH), exist_ok=True)
        os.makedirs(self.LOG_DIR, exist_ok=True)

config = Config()