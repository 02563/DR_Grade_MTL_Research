# src/train.py

import os
import tensorflow as tf
from tensorflow.keras import callbacks
import tensorflow_addons as tfa
from tensorflow.keras.optimizers.schedules import ExponentialDecay
from .model import build_model
from .utils import get_dataset
from .config import config


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
        decay_fn = tf.keras.optimizers.schedules.deserialize(config["decay_fn"])
        return cls(
            initial_lr=config["initial_lr"],
            warmup_steps=config["warmup_steps"],
            decay_fn=decay_fn
        )


class CustomUnfreezeCallback(tf.keras.callbacks.Callback):
    def __init__(self, unfreeze_epoch=10, optimizer=None, loss=None, loss_weights=None, metrics=None):
        super().__init__()
        self.unfreeze_epoch = unfreeze_epoch
        self.optimizer = optimizer
        self.loss = loss
        self.loss_weights = loss_weights
        self.metrics = metrics

    def on_epoch_begin(self, epoch, logs=None):
        if epoch == self.unfreeze_epoch:
            print(f"[回调] 第{self.unfreeze_epoch}轮，解冻 base_model 权重并重新编译模型")
            base_model = self.model.get_layer('resnet50')
            base_model.trainable = True

            # 重新编译模型
            self.model.compile(
                optimizer=self.optimizer,
                loss=self.loss,
                loss_weights=self.loss_weights,
                metrics=self.metrics
            )

            # 重要：强制重新构建训练函数
            self.model.make_train_function()  # 解决 NoneType 错误


def train():
    print("正在启动模型训练...")
    physical_devices = tf.config.list_physical_devices('GPU')
    if physical_devices:
        try:
            tf.config.experimental.set_memory_growth(physical_devices[0], True)
            print("已启用显存动态增长")
        except RuntimeError as e:
            print("配置失败:", e)

    train_dataset = get_dataset('train', config.TRAIN_PARAMS["BATCH_SIZE"])
    val_dataset = get_dataset('val', config.TRAIN_PARAMS["BATCH_SIZE"])

    steps_per_epoch = config.TRAIN_PARAMS["NUM_TRAIN_SAMPLES"] // config.TRAIN_PARAMS["BATCH_SIZE"]
    validation_steps = config.TRAIN_PARAMS["NUM_VAL_SAMPLES"] // config.TRAIN_PARAMS["BATCH_SIZE"]

    print(f"[调试] 训练步数: {steps_per_epoch}")

    total_steps = config.TRAIN_PARAMS["EPOCHS"] * steps_per_epoch
    warmup_steps = config.TRAIN_PARAMS["WARMUP_EPOCHS"] * steps_per_epoch

    decay_fn = ExponentialDecay(
        initial_learning_rate=config.TRAIN_PARAMS["LR"],
        decay_steps=total_steps - warmup_steps,
        decay_rate=0.96
    )

    lr_schedule = WarmUp(
        initial_lr=config.TRAIN_PARAMS["LR"],
        warmup_steps=warmup_steps,
        decay_fn=decay_fn
    )

    optimizer = tf.keras.optimizers.Adam(
        learning_rate=lr_schedule,
        clipnorm=config.TRAIN_PARAMS.get("GRADIENT_CLIP", 1.0)
    )

    model = build_model()

    losses = {
        "grade": "categorical_crossentropy",
        "recon": "mse"
    }
    loss_weights = {
        "grade": 1.0,
        "recon": 0.5
    }
    metrics = {
        "grade": [
            tf.keras.metrics.AUC(name='auc'),
            tf.keras.metrics.CategoricalAccuracy(name='acc')
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
            config.CHECKPOINT_PATH,
            monitor='val_grade_auc',
            save_best_only=True,
            mode='max',
            verbose=1
        ),
        callbacks.TensorBoard(
            log_dir=config.LOG_DIR,
            update_freq='epoch'
        ),
        callbacks.EarlyStopping(
            monitor='val_grade_auc',
            patience=config.TRAIN_PARAMS.get("EARLY_STOPPING_PATIENCE", 10),
            restore_best_weights=True,
            verbose=1
        ),
        CustomUnfreezeCallback(
            unfreeze_epoch=config.TRAIN_PARAMS.get("UNFREEZE_EPOCH", 10),
            optimizer=optimizer,
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


if __name__ == "__main__":
    train()
