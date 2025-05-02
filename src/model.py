# src/model.py （使用 ResNet50 作为主干网络）

import tensorflow as tf
from tensorflow.keras import layers, models
from .config import config

def build_model():
    input_tensor = tf.keras.Input(shape=(224, 224, 3))
    
    # 正确嵌入 base_model，并命名
    base_model = tf.keras.applications.ResNet50(
        include_top=False,
        weights='imagenet',
        input_shape=(224, 224, 3),
        pooling='avg'
    )
    base_model._name = 'resnet50'  # 显式命名
    base_model.trainable = False

    x = base_model(input_tensor)  # ❗注意：使用 base_model(input_tensor)，不是 base_model.output

    # 分类输出
    grade_output = layers.Dense(config.MODEL_PARAMS["NUM_CLASSES"], activation='softmax', name='grade')(x)

    # 重建输出
    x_recon = layers.Dense(7 * 7 * 128, activation='relu')(x)
    x_recon = layers.Reshape((7, 7, 128))(x_recon)
    x_recon = layers.Conv2DTranspose(64, 3, strides=2, padding='same', activation='relu')(x_recon)
    x_recon = layers.Conv2DTranspose(32, 3, strides=2, padding='same', activation='relu')(x_recon)
    x_recon = layers.Conv2DTranspose(3, 3, strides=2, padding='same', activation='sigmoid')(x_recon)
    recon_output = tf.image.resize(x_recon, [224, 224], name='recon')

    # 模型定义
    model = tf.keras.Model(inputs=input_tensor, outputs={"grade": grade_output, "recon": recon_output}, name="dr_mtl_model")

    return model