'''
Author: AlAuMid 2606414786@xiaomi.com
Date: 2025-04-08 00:54:43
LastEditors: AlAuMid 2606414786@xiaomi.com
LastEditTime: 2025-04-19 20:55:22
FilePath: \DR_Grade_MTL_Research\src\model.py
Description: 这是默认设置,请设置`customMade`, 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
'''
import tensorflow as tf
from keras import layers
from tensorflow.keras import Model
from tensorflow.keras import callbacks
from .config import config

class CBAM(layers.Layer):
    """注意力模块（兼容TensorFlow 2.x）"""
    def __init__(self, ratio=8, **kwargs):  # 必须添加**kwargs
        super().__init__(**kwargs)
        self.ratio = ratio

    def build(self, input_shape):
        channels = input_shape[-1]
        self.channel_attention = tf.keras.Sequential([
            layers.GlobalAveragePooling2D(),
            layers.Reshape((1, 1, channels)),
            layers.Conv2D(channels//self.ratio, 1, activation='relu'),
            layers.Conv2D(channels, 1, activation='sigmoid')
        ])
        self.spatial_attention = layers.Conv2D(1, 7, padding='same', activation='sigmoid')
        super().build(input_shape)

    def call(self, inputs):
        # 通道注意力
        ca = self.channel_attention(inputs)
        x = inputs * ca
        # 空间注意力
        sa = self.spatial_attention(x)
        return x * sa
    
    def get_config(self):
        config = super().get_config()
        config.update({"ratio": self.ratio})
        return config

class UnfreezeCallback(callbacks.Callback):
    def on_epoch_begin(self, epoch, logs=None):
        if epoch == 10:
            for layer in self.model.layers:
                if hasattr(layer, 'trainable'):
                    layer.trainable = True

def build_model():
    # 在模型编译前设置混合精度策略
    tf.keras.mixed_precision.set_global_policy('mixed_float16')
    # 主干网络
    base_model = tf.keras.applications.EfficientNetB0(
        weights='imagenet', 
        include_top=False,
        input_shape=(config.IMG_PARAMS["INPUT_SIZE"], config.IMG_PARAMS["INPUT_SIZE"], 3)
    )
    base_model.trainable = False

    for layer in base_model.layers:
        if hasattr(layer, 'trainable'):
            layer.trainable = False  # 或 True
    
    # 特征层
    x = base_model.output
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.5)(x)
    
    # 分类头
    grade_output = layers.Dense(
        config.TASKS['grade']['num_classes'], 
        activation='softmax', 
        name='grade',
        dtype='float32'  # 改为 float32 以避免精度冲突
    )(layers.Dense(128, activation='relu')(x))
    
    model = Model(inputs=base_model.input, outputs=grade_output)
    print("[调试] 模型输入形状:", model.input_shape)
    print("[调试] 模型输出形状:", model.output_shape)
    print("[调试] 模型层信息:", [layer.name for layer in model.layers])
    
    tf.keras.mixed_precision.set_global_policy('mixed_float16')
    return model