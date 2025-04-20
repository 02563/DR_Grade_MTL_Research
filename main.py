'''
Author: AlAuMid 2606414786@xiaomi.com
Date: 2025-04-08 00:55:44
LastEditors: AlAuMid 2606414786@xiaomi.com
LastEditTime: 2025-04-11 19:03:34
FilePath: \DR_Grade_MTL_Research\main.py
Description: 这是默认设置,请设置`customMade`, 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
'''
import sys
import os

# 关键：添加项目根目录到 Python 路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import argparse
from src.train import train
from src.data_prepare import create_tfrecords  # Assuming it exists in this module

def parse_args():
    parser = argparse.ArgumentParser(description="糖尿病性视网膜病变多任务学习")
    parser.add_argument(
        "--prepare_data", 
        action="store_true",
        help="生成TFRecords数据集"
    )
    parser.add_argument(
        "--train",
        action="store_true",
        help="启动模型训练"
    )
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    
    if args.prepare_data:
        print("正在生成TFRecords数据集...")
        try:
            create_tfrecords()
        except Exception as e:
            print(f"数据预处理失败：{e}")
    elif args.train:
        print("正在启动模型训练...")
        try:
            train()
        except Exception as e:
            print(f"模型训练失败：{e}")
    else:
        print("请指定运行模式，例如：")
        print("  python main.py --prepare_data  # 仅预处理数据")
        print("  python main.py --train         # 仅训练模型")