import tensorflow as tf
import numpy as np
from .config import config
import os

def get_dataset(split, batch_size):
    """加载预处理后的数据集"""
    feature_desc = {
        'image': tf.io.FixedLenFeature([], tf.string),
        'grade': tf.io.FixedLenFeature([], tf.int64)
    }
    
    def _parse(example_proto):
        with tf.device('/cpu:0'):
            parsed = tf.io.parse_single_example(example_proto, feature_desc)
            image = tf.io.decode_raw(parsed['image'], tf.uint8)
            image = tf.reshape(image, (config.IMG_PARAMS["INPUT_SIZE"], config.IMG_PARAMS["INPUT_SIZE"], 3))
            image = tf.cast(image, tf.float32) / 255.0

            # 验证
            tf.debugging.assert_rank(image, 3, message="图像张量维度错误")
            tf.debugging.assert_non_negative(image, message="图像像素值为负数")
            tf.debugging.assert_less_equal(image, 1.0, message="图像像素值超过1.0")

            label = tf.one_hot(parsed['grade'], config.TASKS['grade']['num_classes'])
            
            return image, {'grade': label, 'recon': image}  # ⚡改动：输出包含grade标签和recon标签（原图像）

    files = tf.io.matching_files(f"{config.PROCESSED_DIR}/{split}.tfrecords")
    if not tf.io.gfile.exists(f"{config.PROCESSED_DIR}/{split}.tfrecords"):
        raise FileNotFoundError(f"未找到 {split} 数据集的 TFRecords 文件：{config.PROCESSED_DIR}/{split}.tfrecords")
    print(f"[调试] 找到的TFRecords文件：{files.numpy()}")
    print(f"Loading TFRecords from: {config.PROCESSED_DIR}/{split}.tfrecords")

    raw_dataset = tf.data.TFRecordDataset(files)
    dataset = raw_dataset.map(_parse, num_parallel_calls=2)
    dataset = dataset.batch(batch_size, drop_remainder=True)
    dataset = dataset.prefetch(buffer_size=tf.data.AUTOTUNE)

    if split == 'train':
        dataset = dataset.shuffle(1000, reshuffle_each_iteration=True)

    # 简单验证
    test_iter = iter(dataset)
    try:
        first_batch = next(test_iter)
        print(f"[验证] {split}首批次数据形状: 图像{first_batch[0].shape}, 标签{ {k:v.shape for k,v in first_batch[1].items()} }")
    except StopIteration:
        print("[错误] 数据集为空，无法获取批次")

    return dataset
