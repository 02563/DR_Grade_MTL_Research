import argparse
import os
import tensorflow as tf
import json
from src.data_prepare import create_tfrecords
from src.train import train
from src.config import config

def main(args):
    if args.prepare:
        print("正在生成TFRecords数据集...")
        os.makedirs(config.PROCESSED_DIR, exist_ok=True)
        class_weights = create_tfrecords()
        with open(os.path.join(config.PROCESSED_DIR, 'class_weights.json'), 'w') as f:
            json.dump(class_weights, f)
    else:
        class_weights = None

    if args.train:
        print("正在启动模型训练...")
        train()

    if args.test_gpu:
        print("正在测试GPU环境...")
        print(f"TensorFlow版本: {tf.__version__}")
        gpus = tf.config.list_physical_devices('GPU')
        print(f"是否有可用GPU: {gpus}")
        if not gpus:
            print("未检测到GPU，使用CPU运行")
        else:
            # 简单矩阵乘法测试
            import time
            a = tf.random.normal([3000, 3000])
            b = tf.random.normal([3000, 3000])
            start = time.time()
            c = tf.matmul(a, b)
            tf.keras.backend.eval(c)
            duration = time.time() - start
            print(f"矩阵乘法完成，用时 {duration:.2f} 秒")
        print("测试完成。")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="糖尿病性视网膜病变多任务学习项目")
    parser.add_argument("--prepare", action="store_true", help="生成TFRecords文件")
    parser.add_argument("--train", action="store_true", help="训练模型")
    parser.add_argument("--test_gpu", action="store_true", help="测试GPU环境")
    args = parser.parse_args()

    main(args)
