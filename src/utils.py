# src/utils.py （最终修正版）

import tensorflow as tf
import os
import json
from .config import config


def _parse_example(example_proto):
    feature_description = {
        'image': tf.io.FixedLenFeature([], tf.string),
        'grade': tf.io.FixedLenFeature([], tf.int64),
    }
    parsed = tf.io.parse_single_example(example_proto, feature_description)

    label = parsed['grade']  # 不 one-hot，用于计算权重

    # 正确地用 decode_jpeg！
    image = tf.io.decode_jpeg(parsed['image'], channels=3)
    image = tf.image.convert_image_dtype(image, tf.float32)  # [0,1] 浮点归一化

    # recon任务要resize成224（实际上image本来就是224，这里是稳健处理）
    recon = tf.image.resize(image, [config.IMG_PARAMS["INPUT_SIZE"], config.IMG_PARAMS["INPUT_SIZE"]])

    # grade标签
    grade = tf.one_hot(label, depth=config.MODEL_PARAMS['NUM_CLASSES'])

    # 计算 sample weight（以 class_weight 映射）
    with open(os.path.join(config.PROCESSED_DIR, 'class_weights.json'), 'r') as f:
        class_weights = json.load(f)
        # 注意：JSON读入后是字符串键，需转换回整数键
        class_weights = {int(k): v for k, v in class_weights.items()}
    #class_weights = config.TRAIN_PARAMS["CLASS_WEIGHT"]
    weight = tf.cast(tf.gather(list(class_weights.values()), label), tf.float32)

    return image, {"grade": grade, "recon": recon}, {"grade": weight}


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
