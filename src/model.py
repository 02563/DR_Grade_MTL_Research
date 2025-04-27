import tensorflow as tf
from tensorflow.keras import layers, Model, callbacks
from tensorflow.keras import mixed_precision
from .config import config

# 启用混合精度训练
mixed_precision.set_global_policy('mixed_float16')


class CBAM(layers.Layer):
    """CBAM注意力模块（兼容TensorFlow 2.x）"""
    def __init__(self, ratio=8, **kwargs):
        super().__init__(**kwargs)
        self.ratio = ratio

    def build(self, input_shape):
        channels = input_shape[-1]
        self.channel_attention = tf.keras.Sequential([
            layers.GlobalAveragePooling2D(),
            layers.Reshape((1, 1, channels)),
            layers.Conv2D(channels // self.ratio, 1, activation='relu'),
            layers.Conv2D(channels, 1, activation='sigmoid')
        ])
        self.spatial_attention = layers.Conv2D(1, 7, padding='same', activation='sigmoid')
        super().build(input_shape)

    def call(self, inputs):
        ca = self.channel_attention(inputs)
        x = inputs * ca
        sa = self.spatial_attention(x)
        return x * sa

    def get_config(self):
        config = super().get_config()
        config.update({"ratio": self.ratio})
        return config


class UnfreezeCallback(callbacks.Callback):
    """第10轮开始解冻base_model"""
    def on_epoch_begin(self, epoch, logs=None):
        if epoch == 10:
            self.model.base_model.trainable = True
            print("[回调] 第10轮，解冻 base_model 参数！")


def build_model():
    """构建多任务模型（分类+重建）"""
    base_model = tf.keras.applications.EfficientNetB0(
        weights='imagenet',
        include_top=False,
        input_shape=(config.IMG_PARAMS["INPUT_SIZE"], config.IMG_PARAMS["INPUT_SIZE"], 3)
    )
    base_model.trainable = False

    x = base_model.output
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.5)(x)

    # 分类分支
    grade_fc = layers.Dense(128, activation='relu')(x)
    grade_output = layers.Dense(
        config.TASKS['grade']['num_classes'],
        activation='softmax',
        name='grade',
        dtype='float32'  # 强制输出float32，防止混合精度出错
    )(grade_fc)

    # 重建分支
    recon_fc = layers.Dense(7 * 7 * 32, activation='relu')(x)
    recon_fc = layers.Reshape((7, 7, 32))(recon_fc)
    recon_fc = layers.Conv2DTranspose(32, 3, strides=2, padding='same', activation='relu')(recon_fc)
    recon_fc = layers.Conv2DTranspose(16, 3, strides=2, padding='same', activation='relu')(recon_fc)
    recon_fc = layers.Conv2DTranspose(3, 3, strides=2, padding='same', activation='sigmoid')(recon_fc)
    recon_output = tf.image.resize(
        recon_fc,
        size=(config.IMG_PARAMS["INPUT_SIZE"], config.IMG_PARAMS["INPUT_SIZE"]),
        method='bilinear',
        name="recon"
    )

    model = Model(inputs=base_model.input, outputs={"grade": grade_output, "recon": recon_output})
    model.base_model = base_model  # 保存base_model方便后续callback使用

    print("[调试] 模型输入形状:", model.input_shape)
    print("[调试] 模型输出形状:", {k: v.shape for k, v in model.output.items()})
    print("[调试] 模型层信息:", [layer.name for layer in model.layers])

    return model
