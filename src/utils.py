# src/utils.py （最终修正版）

import tensorflow as tf
import os
from .config import config


def _parse_example(example_proto):
    feature_description = {
        'image': tf.io.FixedLenFeature([], tf.string),
        'grade': tf.io.FixedLenFeature([], tf.int64),
    }
    parsed = tf.io.parse_single_example(example_proto, feature_description)

    # 正确地用 decode_jpeg！
    image = tf.io.decode_jpeg(parsed['image'], channels=3)
    image = tf.image.convert_image_dtype(image, tf.float32)  # [0,1] 浮点归一化

    # recon任务要resize成224（实际上image本来就是224，这里是稳健处理）
    recon = tf.image.resize(image, [config.IMG_PARAMS["INPUT_SIZE"], config.IMG_PARAMS["INPUT_SIZE"]])

    # grade标签
    grade = tf.one_hot(parsed['grade'], depth=config.MODEL_PARAMS["NUM_CLASSES"])

    return image, {"grade": grade, "recon": recon}


def get_dataset(split='train', batch_size=32):
    """读取 TFRecord，返回Dataset"""
    tfrecord_path = os.path.join(config.PROCESSED_DIR, f"{split}.tfrecords")
    print(f"Loading TFRecords from: {tfrecord_path}")

    raw_dataset = tf.data.TFRecordDataset(tfrecord_path)
    dataset = raw_dataset.map(_parse_example, num_parallel_calls=tf.data.AUTOTUNE)

    if split == 'train':
        dataset = dataset.shuffle(1024)
        dataset = dataset.repeat()   # 数据循环
    
    dataset = dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    return dataset
