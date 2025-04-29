import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

def build_model():
    """构建多任务学习模型：分类 + 图像重建"""

    # EfficientNetB0作为主干
    base_model = keras.applications.EfficientNetB0(
        include_top=False,
        input_shape=(224, 224, 3),
        weights='imagenet',
        pooling='avg'
    )
    base_model.trainable = False

    inputs = layers.Input(shape=(224, 224, 3))
    # 归一化
    x = layers.Rescaling(1.0 / 255)(inputs)
    x = layers.Normalization()(x)
    x = base_model(x)

    # 任务1：DR分级分类
    grade_output = layers.Dense(5, activation='softmax', name='grade')(x)

    # 任务2：图像重建（decoder）
    x_recon = layers.Dense(7 * 7 * 128, activation='relu')(x)
    x_recon = layers.Reshape((7, 7, 128))(x_recon)
    x_recon = layers.Conv2DTranspose(64, 3, strides=2, padding='same', activation='relu')(x_recon)
    x_recon = layers.Conv2DTranspose(32, 3, strides=2, padding='same', activation='relu')(x_recon)
    x_recon = layers.Conv2DTranspose(3, 3, strides=2, padding='same', activation='sigmoid')(x_recon)

    # 输出
    recon_output = tf.image.resize(x_recon, size=(224, 224), method="bilinear", name='recon')

    model = keras.Model(inputs=inputs, outputs={"grade": grade_output, "recon": recon_output})

    # 打印结构调试
    print("[调试] 模型输入形状:", inputs.shape)
    print("[调试] 模型输出形状:", {k: v.shape for k, v in model.output.items()})
    print("[调试] 模型层信息:", [layer.name for layer in model.layers])

    return model
