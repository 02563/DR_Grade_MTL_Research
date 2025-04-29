import os
import cv2
import pandas as pd
import numpy as np
import tensorflow as tf
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from .config import config

def preprocess_image(image_path, augment=True):
    """视网膜图像预处理，生成标准尺寸"""
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"无法读取图像: {image_path}")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # 保证足够大
    h, w = img.shape[:2]
    scale = max(config.IMG_PARAMS["CROP_SIZE"] / h, config.IMG_PARAMS["CROP_SIZE"] / w)
    if scale > 1:
        img = cv2.resize(img, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_LINEAR)

    # CLAHE增强
    lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    l = clahe.apply(l)
    lab = cv2.merge((l, a, b))
    img = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)

    # 中心裁剪
    h, w = img.shape[:2]
    start_x = (w - config.IMG_PARAMS["CROP_SIZE"]) // 2
    start_y = (h - config.IMG_PARAMS["CROP_SIZE"]) // 2
    img = img[start_y:start_y+config.IMG_PARAMS["CROP_SIZE"],
              start_x:start_x+config.IMG_PARAMS["CROP_SIZE"]]

    # 再 resize 到标准输入尺寸224
    img = cv2.resize(img, (config.IMG_PARAMS["INPUT_SIZE"], config.IMG_PARAMS["INPUT_SIZE"]))

    return img

def create_tfrecords():
    """生成TFRecords并记录样本数"""
    assert os.path.exists(config.CSV_PATH), f"CSV文件不存在于 {config.CSV_PATH}"

    df = pd.read_csv(config.CSV_PATH)
    def find_image_path(id_code):
        png_path = os.path.join(config.RAW_DIR, "train_images", f"{id_code}.png")
        jpeg_path = os.path.join(config.RAW_DIR, "train_images", f"{id_code}.jpeg")
        if os.path.exists(png_path):
            return png_path
        elif os.path.exists(jpeg_path):
            return jpeg_path
        else:
            return None  # 之后统一排除不存在的

    df['image_path'] = df['id_code'].apply(find_image_path)

    missing_files = [p for p in df['image_path'] if not os.path.exists(p)]
    if missing_files:
        raise FileNotFoundError(f"缺失 {len(missing_files)} 个图像文件")

    train_df, val_df = train_test_split(
        df, test_size=config.TRAIN_PARAMS["VAL_SPLIT"],
        stratify=df['diagnosis'], random_state=42
    )

    invalid_samples = []

    def _write_dataset(df, split):
        valid_count = 0
        with tf.io.TFRecordWriter(f"{config.PROCESSED_DIR}/{split}.tfrecords") as writer:
            for _, row in tqdm(df.iterrows(), total=len(df), desc=f"生成 {split} 数据集"):
                try:
                    img = preprocess_image(row['image_path'], augment=(split == 'train'))
                    img_bytes = tf.io.encode_jpeg(img).numpy()
                    example = tf.train.Example(features=tf.train.Features(feature={
                        'image': tf.train.Feature(bytes_list=tf.train.BytesList(value=[img_bytes])),
                        'grade': tf.train.Feature(int64_list=tf.train.Int64List(value=[row['diagnosis']]))
                    }))
                    writer.write(example.SerializeToString())
                    valid_count += 1
                except Exception as e:
                    invalid_samples.append(f"{row['id_code']}: {str(e)}")
        return valid_count

    config.TRAIN_PARAMS["NUM_TRAIN_SAMPLES"] = _write_dataset(train_df, "train")
    config.TRAIN_PARAMS["NUM_VAL_SAMPLES"] = _write_dataset(val_df, "val")

    assert config.TRAIN_PARAMS["NUM_TRAIN_SAMPLES"] > 0, "训练集样本数为0！"
    assert config.TRAIN_PARAMS["NUM_VAL_SAMPLES"] > 0, "验证集样本数为0！"

    print(f"[调试] 训练集有效样本数：{config.TRAIN_PARAMS['NUM_TRAIN_SAMPLES']}")
    print(f"[调试] 验证集有效样本数：{config.TRAIN_PARAMS['NUM_VAL_SAMPLES']}")
    print(f"[调试] 生成的训练集文件路径：{config.PROCESSED_DIR}/train.tfrecords")
    print(f"[调试] 生成的验证集文件路径：{config.PROCESSED_DIR}/val.tfrecords")

    if invalid_samples:
        print(f"跳过 {len(invalid_samples)} 个无效样本")
        with open("./invalid_samples.txt", "w") as f:
            f.write("\n".join(invalid_samples))
