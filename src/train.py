import os
import tensorflow as tf
import json
from tensorflow.keras import callbacks
from tensorflow.keras.optimizers.schedules import CosineDecay
from tensorflow_addons.optimizers import AdamW
from tensorflow.keras.losses import CategoricalCrossentropy
from tensorflow.keras import mixed_precision
from tensorflow.keras.utils import register_keras_serializable
from tensorflow.keras.mixed_precision import LossScaleOptimizer
from .model import build_model
from .utils import get_dataset
from .config import config

mixed_precision.set_global_policy('mixed_float16')

@register_keras_serializable()
class WarmUp(tf.keras.optimizers.schedules.LearningRateSchedule):
    def __init__(self, initial_lr, warmup_steps, decay_fn):
        super().__init__()
        self.initial_lr = initial_lr
        self.warmup_steps = warmup_steps
        self.decay_fn = decay_fn

    def __call__(self, step):
        step = tf.cast(step, tf.float32)
        warmup_lr = self.initial_lr * (step / self.warmup_steps)
        decay_lr = self.decay_fn(step - self.warmup_steps)
        return tf.cond(step < self.warmup_steps, lambda: warmup_lr, lambda: decay_lr)

    def get_config(self):
        return {
            "initial_lr": self.initial_lr,
            "warmup_steps": self.warmup_steps,
            "decay_fn": tf.keras.optimizers.schedules.serialize(self.decay_fn)
        }

    @classmethod
    def from_config(cls, config):
        config["decay_fn"] = tf.keras.optimizers.schedules.deserialize(config["decay_fn"])
        return cls(**config)

class CustomUnfreezeCallback(tf.keras.callbacks.Callback):
    def __init__(self, unfreeze_epoch=5, loss=None, loss_weights=None, metrics=None):
        super().__init__()
        self.unfreeze_epoch = unfreeze_epoch
        self.loss = loss
        self.loss_weights = loss_weights
        self.metrics = metrics

    def on_epoch_begin(self, epoch, logs=None):
        if epoch == self.unfreeze_epoch:
            print(f"[回调] 第{self.unfreeze_epoch}轮，解冻 ResNet 后期 block 并重新编译模型")
            unfrozen = 0
            for layer in self.model.layers:
                if any(x in layer.name for x in ["conv4", "conv5"]):
                    layer.trainable = True
                    unfrozen += 1
            print(f"[回调] 解冻了 {unfrozen} 层（conv4/conv5）")

            from tensorflow_addons.optimizers import AdamW
            from tensorflow.keras.mixed_precision import LossScaleOptimizer
            from src.config import config

            new_optimizer = AdamW(
                learning_rate=self.model.optimizer.learning_rate,
                weight_decay=config.MODEL_PARAMS["WEIGHT_DECAY"],
                clipnorm=config.TRAIN_PARAMS["GRADIENT_CLIP"]
            )
            new_optimizer = LossScaleOptimizer(new_optimizer)

            self.model.compile(
                optimizer=new_optimizer,
                loss=self.loss,
                loss_weights=self.loss_weights,
                metrics=self.metrics
            )
            self.model.make_train_function()

def focal_loss(gamma=2., alpha=0.25):
    def loss(y_true, y_pred):
        y_true = tf.cast(y_true, tf.float32)
        y_pred = tf.cast(y_pred, tf.float32)  # 强制转换解决 float16/32 不一致问题
        epsilon = tf.keras.backend.epsilon()
        y_pred = tf.clip_by_value(y_pred, epsilon, 1. - epsilon)
        cross_entropy = -y_true * tf.math.log(y_pred)
        weight = alpha * tf.pow(1 - y_pred, gamma)
        return tf.reduce_sum(weight * cross_entropy, axis=-1)
    return loss

def train():
    with open(os.path.join(config.PROCESSED_DIR, "class_weights.json"), "r") as f:
        class_weights = json.load(f)

    def compute_sample_weights(labels, class_weights):
        label_indices = tf.argmax(labels, axis=1)
        weights = tf.gather([class_weights[str(i)] for i in range(len(class_weights))], label_indices)
        return tf.cast(weights, tf.float32)

    def add_sample_weights(x, y, sw=None):
        grade_labels = y["grade"]
        weights = compute_sample_weights(grade_labels, class_weights)
        return x, y, {"grade": weights, "recon": None}

    print("正在启动模型训练...")
    physical_devices = tf.config.list_physical_devices('GPU')
    if physical_devices:
        try:
            tf.config.experimental.set_memory_growth(physical_devices[0], True)
            print("已启用显存动态增长")
        except RuntimeError as e:
            print("配置失败:", e)

    train_dataset = get_dataset('train', config.TRAIN_PARAMS["BATCH_SIZE"])
    train_dataset = train_dataset.map(add_sample_weights, num_parallel_calls=tf.data.AUTOTUNE)
    val_dataset = get_dataset('val', config.TRAIN_PARAMS["BATCH_SIZE"])
    val_dataset = val_dataset.map(add_sample_weights, num_parallel_calls=tf.data.AUTOTUNE)

    steps_per_epoch = config.TRAIN_PARAMS["NUM_TRAIN_SAMPLES"] // config.TRAIN_PARAMS["BATCH_SIZE"]
    validation_steps = config.TRAIN_PARAMS["NUM_VAL_SAMPLES"] // config.TRAIN_PARAMS["BATCH_SIZE"]

    total_steps = config.TRAIN_PARAMS["EPOCHS"] * steps_per_epoch
    warmup_steps = config.TRAIN_PARAMS["WARMUP_EPOCHS"] * steps_per_epoch

    decay_fn = CosineDecay(
        initial_learning_rate=config.TRAIN_PARAMS["LR"],
        decay_steps=total_steps - warmup_steps,
        alpha=1e-5
    )

    lr_schedule = WarmUp(
        initial_lr=config.TRAIN_PARAMS["LR"],
        warmup_steps=warmup_steps,
        decay_fn=decay_fn
    )

    optimizer = AdamW(
        learning_rate=lr_schedule,
        weight_decay=config.MODEL_PARAMS["WEIGHT_DECAY"],
        clipnorm=config.TRAIN_PARAMS["GRADIENT_CLIP"]
    )
    optimizer = LossScaleOptimizer(optimizer)

    model = build_model(
        input_shape=(224, 224, 3),
        num_classes=config.MODEL_PARAMS["NUM_CLASSES"],
        dropout_rate=config.MODEL_PARAMS["DROPOUT"],
        weight_decay=config.MODEL_PARAMS["WEIGHT_DECAY"]
    )

    losses = {
        "grade": focal_loss(gamma=2.0, alpha=0.25),
        "recon": "mse"
    }
    loss_weights = {
        "grade": 1.0,
        "recon": 0.001  # 降低重建权重
    }
    metrics = {
        "grade": [
            tf.keras.metrics.AUC(name='grade_auc'),
            tf.keras.metrics.CategoricalAccuracy(name='grade_acc')
        ],
        "recon": []
    }

    model.compile(
        optimizer=optimizer,
        loss=losses,
        loss_weights=loss_weights,
        metrics=metrics
    )

    callbacks_list = [
        callbacks.ModelCheckpoint(
            filepath=config.CHECKPOINT_PATH,
            monitor='val_grade_grade_auc',
            save_best_only=True,
            mode='max',
            verbose=1,
            save_format='tf'
        ),
        callbacks.TensorBoard(
            log_dir=config.LOG_DIR,
            update_freq='epoch'
        ),
        callbacks.EarlyStopping(
            monitor='val_grade_grade_auc',
            patience=10,
            restore_best_weights=True,
            verbose=1
        ),
        CustomUnfreezeCallback(
            unfreeze_epoch=config.TRAIN_PARAMS["UNFREEZE_EPOCH"],
            loss=losses,
            loss_weights=loss_weights,
            metrics=metrics
        )
    ]

    history = model.fit(
        train_dataset,
        epochs=config.TRAIN_PARAMS["EPOCHS"],
        steps_per_epoch=steps_per_epoch,
        validation_data=val_dataset,
        validation_steps=validation_steps,
        callbacks=callbacks_list
    )
    return history
