import os
import tensorflow as tf
from tensorflow.keras.mixed_precision import experimental as mixed_precision

# 适配 TensorFlow 2.8 的混合精度设置
policy = mixed_precision.Policy('mixed_float16')
mixed_precision.set_policy(policy)

from tensorflow.keras import callbacks
import tensorflow_addons as tfa
from tensorflow.keras.optimizers.schedules import CosineDecay
from .model import build_model, UnfreezeCallback
from .utils import get_dataset
from .config import config

class WarmUp(tf.keras.optimizers.schedules.LearningRateSchedule):
    """WarmUp + 衰减联合学习率策略"""
    def __init__(self, initial_lr, warmup_steps, decay_fn):
        super().__init__()
        self.initial_lr = initial_lr
        self.warmup_steps = warmup_steps
        self.decay_fn = decay_fn

    def __call__(self, step):
        step = tf.cast(step, tf.float32)
        warmup_lr = self.initial_lr * (step / self.warmup_steps)
        decay_lr = self.decay_fn(step - self.warmup_steps)
        return tf.cond(
            step < self.warmup_steps,
            lambda: warmup_lr,
            lambda: decay_lr
        )

    def get_config(self):
        return {
            "initial_lr": self.initial_lr,
            "warmup_steps": self.warmup_steps,
            "decay_fn": tf.keras.optimizers.schedules.serialize(self.decay_fn)
        }

    @classmethod
    def from_config(cls, config):
        decay_fn = tf.keras.optimizers.schedules.deserialize(config["decay_fn"])
        return cls(
            initial_lr=config["initial_lr"],
            warmup_steps=config["warmup_steps"],
            decay_fn=decay_fn
        )

def train():
    """训练模型"""
    # GPU配置
    physical_devices = tf.config.list_physical_devices('GPU')
    if physical_devices:
        try:
            tf.config.experimental.set_memory_growth(physical_devices[0], True)
            print("已启用显存动态增长")
        except RuntimeError as e:
            print("配置失败:", e)

    # 准备数据
    train_dataset = get_dataset('train', config.TRAIN_PARAMS["BATCH_SIZE"])
    val_dataset = get_dataset('val', config.TRAIN_PARAMS["BATCH_SIZE"])

    steps_per_epoch = config.TRAIN_PARAMS["NUM_TRAIN_SAMPLES"] // config.TRAIN_PARAMS["BATCH_SIZE"]
    validation_steps = config.TRAIN_PARAMS["NUM_VAL_SAMPLES"] // config.TRAIN_PARAMS["BATCH_SIZE"]

    print(f"[调试] 训练步数: {steps_per_epoch}, 验证步数: {validation_steps}")

    # 学习率调度器
    total_steps = config.TRAIN_PARAMS["EPOCHS"] * steps_per_epoch
    warmup_steps = config.TRAIN_PARAMS["WARMUP_EPOCHS"] * steps_per_epoch
    decay_fn = CosineDecay(
        initial_learning_rate=config.TRAIN_PARAMS["LR"],
        decay_steps=total_steps - warmup_steps
    )
    lr_schedule = WarmUp(
        initial_lr=config.TRAIN_PARAMS["LR"],
        warmup_steps=warmup_steps,
        decay_fn=decay_fn
    )

    optimizer = tf.keras.optimizers.Adam(
        learning_rate=lr_schedule,
        clipnorm=config.TRAIN_PARAMS["GRADIENT_CLIP"]
    )

    # 构建模型
    model = build_model()

    # 编译模型
    model.compile(
        optimizer=optimizer,
        loss={
            "grade": "categorical_crossentropy",
            "recon": "mse"
        },
        loss_weights={
            "grade": 1.0,
            "recon": 0.5
        },
        metrics={
            "grade": [
                tf.keras.metrics.AUC(name='auc'),
                tf.keras.metrics.CategoricalAccuracy(name='acc')
            ],
            "recon": []
        }
    )

    # 回调函数
    callbacks_list = [
        callbacks.ModelCheckpoint(
            config.CHECKPOINT_PATH,
            monitor='val_grade_auc',
            save_best_only=True,
            mode='max'
        ),
        callbacks.TensorBoard(config.LOG_DIR),
        callbacks.EarlyStopping(monitor='val_grade_auc', patience=10, restore_best_weights=True),
        UnfreezeCallback()
    ]

    # 开始训练
    history = model.fit(
        train_dataset,
        epochs=config.TRAIN_PARAMS["EPOCHS"],
        validation_data=val_dataset,
        callbacks=callbacks_list
    )
    return history

if __name__ == "__main__":
    train()
