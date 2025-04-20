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
            image = tf.io.decode_raw(parsed['image'], tf.uint8)  # 修改为 uint8
            image = tf.reshape(image, (config.IMG_PARAMS["INPUT_SIZE"], 
                            config.IMG_PARAMS["INPUT_SIZE"], 3))
            print(f"[调试] 图像形状: {image.shape}")  # 确保输出为(224, 224, 3)
            image = tf.cast(image, tf.float32) / 255.0  # 在GPU上转换

            # 验证图像数据
            tf.debugging.assert_rank(image, 3, message="图像张量维度错误")
            tf.debugging.assert_non_negative(image, message="图像像素值为负数")
            tf.debugging.assert_less_equal(image, 1.0, message="图像像素值超过1.0")

            label = tf.one_hot(parsed['grade'], config.TASKS['grade']['num_classes'])
            
            # 添加调试输出
            tf.debugging.assert_non_negative(image, message="图像像素值异常")
            tf.debugging.assert_less_equal(image, 1.0, message="图像像素值超过1.0")
            return image, label

    # 匹配 TFRecords 文件
    files = tf.io.matching_files(f"{config.PROCESSED_DIR}/{split}.tfrecords")
    if not tf.io.gfile.exists(f"{config.PROCESSED_DIR}/{split}.tfrecords"):
        raise FileNotFoundError(f"未找到 {split} 数据集的 TFRecords 文件：{config.PROCESSED_DIR}/{split}.tfrecords")
    print(f"[调试] 找到的TFRecords文件：{files.numpy()}")
    print(f"Loading TFRecords from: {config.PROCESSED_DIR}/{split}.tfrecords")

    # 创建数据集
    raw_dataset = tf.data.TFRecordDataset(files)
    dataset = raw_dataset.map(_parse, num_parallel_calls=2)
    dataset = dataset.batch(batch_size, drop_remainder=True) 
    # 批量化并验证数据集
    
    dataset = dataset.prefetch(buffer_size=tf.data.AUTOTUNE)  # 自动调整预取量
    
    if split == 'train':
        dataset = dataset.shuffle(1000, reshuffle_each_iteration=True)
        dataset = dataset.repeat()

    '''
    # 验证数据集是否正确
    def _debug_validate_batch(dataset):
        # 创建数据集的副本进行验证
        test_ds = dataset.take(1)
        for batch in test_ds:
            images, labels = batch
            assert images.shape[1:] == (config.IMG_PARAMS["INPUT_SIZE"], config.IMG_PARAMS["INPUT_SIZE"], 3), \
                f"图像形状错误: 实际 {images.shape}, 预期 ({config.IMG_PARAMS['INPUT_SIZE']}, {config.IMG_PARAMS['INPUT_SIZE']}, 3)"
            assert tf.reduce_all(images >= 0.0) and tf.reduce_all(images <= 1.0), \
                "图像像素值应在[0,1]范围内"
            label_values = tf.argmax(labels, axis=1)
            assert tf.reduce_all(label_values >= 0) and tf.reduce_all(label_values < config.TASKS['grade']['num_classes']), \
                f"标签值越界: 应在0-{config.TASKS['grade']['num_classes']-1}之间"
            print(f"[验证通过] 图像形状: {images.shape}, 标签形状: {labels.shape}")
        return dataset
    '''


    #print(f"[调试] {split}数据集样本总数: {len(list(dataset))}")  # 确保不为零

    test_iter = iter(dataset)
    try:
        first_batch = next(test_iter)
        print(f"[验证] 第一个批次数据形状: {first_batch[0].shape}, 标签形状: {first_batch[1].shape}")
    except StopIteration:
        print("[错误] 数据集为空，无法获取批次")

    # 返回数据集
    return dataset