import tensorflow as tf
from tensorflow.keras import layers, models, regularizers
from tensorflow.keras.applications import ResNet50

class CBAM(tf.keras.layers.Layer):
    def __init__(self, reduction_ratio=16, kernel_size=7, name=None):
        super(CBAM, self).__init__(name=name)
        self.reduction_ratio = reduction_ratio
        self.kernel_size = kernel_size

    def build(self, input_shape):
        channel = input_shape[-1]
        self.shared_mlp = tf.keras.Sequential([
            layers.Dense(channel // self.reduction_ratio, activation='relu', use_bias=True),
            layers.Dense(channel, use_bias=True)
        ])
        self.conv_spatial = layers.Conv2D(filters=1, kernel_size=self.kernel_size,
                                          strides=1, padding='same', activation='sigmoid')

    def call(self, inputs):
        avg_pool = tf.reduce_mean(inputs, axis=[1, 2], keepdims=True)
        max_pool = tf.reduce_max(inputs, axis=[1, 2], keepdims=True)
        channel_attention = tf.nn.sigmoid(self.shared_mlp(avg_pool) + self.shared_mlp(max_pool))
        x = inputs * channel_attention

        avg_pool = tf.reduce_mean(x, axis=-1, keepdims=True)
        max_pool = tf.reduce_max(x, axis=-1, keepdims=True)
        concat = tf.concat([avg_pool, max_pool], axis=-1)
        spatial_attention = self.conv_spatial(concat)
        x = x * spatial_attention
        return x

    def get_config(self):
        config = super().get_config()
        config.update({
            "reduction_ratio": self.reduction_ratio,
            "kernel_size": self.kernel_size,
        })
        return config

def conv_block(x, filters, name):
    x = layers.Conv2D(filters, 3, padding='same', activation='relu', name=name+'_conv1')(x)
    x = layers.BatchNormalization(name=name+'_bn1')(x)
    x = layers.Conv2D(filters, 3, padding='same', activation='relu', name=name+'_conv2')(x)
    x = layers.BatchNormalization(name=name+'_bn2')(x)
    return x

def build_model(input_shape=(224, 224, 3), num_classes=5, dropout_rate=0.5, weight_decay=1e-4, use_cbam=True):
    inputs = layers.Input(shape=input_shape)
    base_model = ResNet50(include_top=False, weights='imagenet', input_tensor=inputs)

    skip1 = base_model.get_layer("conv2_block3_out").output  # 56x56x256
    skip2 = base_model.get_layer("conv3_block4_out").output  # 28x28x512
    skip3 = base_model.get_layer("conv4_block6_out").output  # 14x14x1024
    x = base_model.get_layer("conv5_block3_out").output       # 7x7x2048

    if use_cbam:
        skip2 = CBAM(name="cbam2")(skip2)
        skip3 = CBAM(name="cbam3")(skip3)
        x = CBAM(name="cbam4")(x)

    x_pool = layers.GlobalAveragePooling2D()(x)
    x_class = layers.Dense(256, activation='relu',
        kernel_regularizer=regularizers.l2(weight_decay))(x_pool)
    x_class = layers.BatchNormalization()(x_class)
    x_class = layers.Dropout(dropout_rate)(x_class)

    x_class = layers.Dense(128, activation='relu',
        kernel_regularizer=regularizers.l2(weight_decay))(x_class)
    x_class = layers.BatchNormalization()(x_class)
    x_class = layers.Dropout(dropout_rate)(x_class)

    x_class = layers.Dense(64, activation='relu',
        kernel_regularizer=regularizers.l2(weight_decay))(x_class)
    x_class = layers.BatchNormalization()(x_class)
    x_class = layers.Dropout(dropout_rate)(x_class)

    grade_output = layers.Dense(num_classes, activation='softmax', name='grade')(x_class)

    # UNet-style decoder with skip connections
    x = layers.Conv2DTranspose(1024, 3, strides=2, padding='same', activation='relu')(x)
    x = layers.Concatenate()([x, skip3])
    x = conv_block(x, 512, name='up1')

    x = layers.Conv2DTranspose(512, 3, strides=2, padding='same', activation='relu')(x)
    x = layers.Concatenate()([x, skip2])
    x = conv_block(x, 256, name='up2')

    x = layers.Conv2DTranspose(256, 3, strides=2, padding='same', activation='relu')(x)
    x = layers.Concatenate()([x, skip1])
    x = conv_block(x, 128, name='up3')

    x = layers.Conv2DTranspose(64, 3, strides=2, padding='same', activation='relu')(x)
    x = conv_block(x, 64, name='up4')

    recon_output = layers.Conv2D(3, 1, activation='sigmoid', name='recon')(x)
    recon_output = tf.image.resize(recon_output, [224, 224])

    model = models.Model(inputs=inputs, outputs={"grade": grade_output, "recon": recon_output})
    return model
