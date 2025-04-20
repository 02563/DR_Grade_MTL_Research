import tensorflow as tf
from tensorflow.keras import callbacks, optimizers
import tensorflow_addons as tfa
from tensorflow.keras.optimizers.schedules import ExponentialDecay
from .model import build_model
from .utils import get_dataset
from .config import config

class WarmUp(tf.keras.optimizers.schedules.LearningRateSchedule):
    """学习率Warmup策略"""
    def __init__(self, initial_lr, warmup_steps, decay_fn):
        super().__init__()
        self.initial_lr = initial_lr
        self.warmup_steps = warmup_steps
        self.decay_fn = decay_fn

    def __call__(self, step):
        step = tf.cast(step, tf.float32)  # 确保 step 是浮点数
        warmup_lr = self.initial_lr * (step / tf.cast(self.warmup_steps, tf.float32))
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

def focal_loss(gamma=2.0, alpha=0.25):
    def loss_fn(y_true, y_pred):
        y_pred = tf.clip_by_value(y_pred, 1e-7, 1.0 - 1e-7)
        ce = -y_true * tf.math.log(y_pred)
        weight = alpha * tf.pow(1.0 - y_pred, gamma)
        return tf.reduce_mean(tf.reduce_sum(weight * ce, axis=-1))
    return loss_fn

class UnfreezeCallback(callbacks.Callback):
    def on_epoch_begin(self, epoch, logs=None):
        if epoch == 10:
            # 直接解冻主干网络的所有层（假设主干网络是模型的第二层）
            base_model = self.model.layers[1]  # 确保此处索引正确
            base_model.trainable = True  # 解冻全部层

def _validate_dataset(dataset, split_name, expected_batch_size):
    """验证数据集是否包含有效批次"""
    try:
        # 检查数据集是否为空
        first_batch = next(iter(dataset))
        print(f"\n[验证] {split_name}数据集首个批次检查通过")
    except StopIteration:
        raise ValueError(f"{split_name}数据集为空，无法获取任何批次")

    # 检查图像和标签形状
    images, labels = first_batch
    expected_image_shape = (expected_batch_size, 
                           config.IMG_PARAMS["INPUT_SIZE"], 
                           config.IMG_PARAMS["INPUT_SIZE"], 
                           3)
    expected_label_shape = (expected_batch_size, 
                           config.TASKS['grade']['num_classes'])
    
    assert images.shape == expected_image_shape, (
        f"{split_name}图像形状错误: 实际 {images.shape}, 预期 {expected_image_shape}"
    )
    assert labels.shape == expected_label_shape, (
        f"{split_name}标签形状错误: 实际 {labels.shape}, 预期 {expected_label_shape}"
    )

    # 检查像素值范围
    assert tf.reduce_min(images) >= 0.0 and tf.reduce_max(images) <= 1.0, (
        f"{split_name}像素值异常: 应在[0, 1]范围内"
    )

    # 检查标签类别有效性
    label_values = tf.argmax(labels, axis=1)
    assert tf.reduce_min(label_values) >= 0 and tf.reduce_max(label_values) < config.TASKS['grade']['num_classes'], (
        f"{split_name}标签值越界: 应在0-{config.TASKS['grade']['num_classes']-1}之间"
    )
    print(f"[验证] {split_name}数据集检查全部通过")

def train():
    physical_devices = tf.config.list_physical_devices('GPU')
    if physical_devices:
        try:
            # 启用显存动态增长（删除硬性限制）
            tf.config.experimental.set_memory_growth(physical_devices[0], True)
            print("已启用显存动态增长")
        except RuntimeError as e:
            print("配置失败:", e)
            
    # 数据加载
    train_dataset = get_dataset('train', config.TRAIN_PARAMS["BATCH_SIZE"])
    val_dataset = get_dataset('val', config.TRAIN_PARAMS["BATCH_SIZE"])


    # 添加数据检查
    _validate_dataset(train_dataset, "训练集", config.TRAIN_PARAMS["BATCH_SIZE"])
    _validate_dataset(val_dataset, "验证集", config.TRAIN_PARAMS["BATCH_SIZE"])
    
    # 计算训练步数
    #steps_per_epoch = config.TRAIN_PARAMS["NUM_TRAIN_SAMPLES"] // config.TRAIN_PARAMS["BATCH_SIZE"]
    #validation_steps = config.TRAIN_PARAMS["NUM_VAL_SAMPLES"] // config.TRAIN_PARAMS["BATCH_SIZE"]
    
    
    # （向上取整）
    #steps_per_epoch = (config.TRAIN_PARAMS["NUM_TRAIN_SAMPLES"] + config.TRAIN_PARAMS["BATCH_SIZE"] - 1) // config.TRAIN_PARAMS["BATCH_SIZE"]
    #validation_steps = (config.TRAIN_PARAMS["NUM_VAL_SAMPLES"] + config.TRAIN_PARAMS["BATCH_SIZE"] - 1) // config.TRAIN_PARAMS["BATCH_SIZE"]

    # 修改为向下取整：
    steps_per_epoch = config.TRAIN_PARAMS["NUM_TRAIN_SAMPLES"] // config.TRAIN_PARAMS["BATCH_SIZE"]
    validation_steps = config.TRAIN_PARAMS["NUM_VAL_SAMPLES"] // config.TRAIN_PARAMS["BATCH_SIZE"]

    print(f"[调试] 训练步数: {steps_per_epoch}, 验证步数: {validation_steps}")
    print("NUM_TRAIN_SAMPLES:", config.TRAIN_PARAMS["NUM_TRAIN_SAMPLES"])
    print("BATCH_SIZE:", config.TRAIN_PARAMS["BATCH_SIZE"])
    
    
    # 学习率调度
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

    # 调试输出
    for step in range(10):
        print(f"Step {step}: Learning rate = {lr_schedule(step).numpy()}")
    
    # 优化器
    optimizer = tf.keras.optimizers.Adam(
        learning_rate=lr_schedule,
        clipnorm=config.TRAIN_PARAMS["GRADIENT_CLIP"]
    )
    
    # 编译模型
    model = build_model()
    model.compile(
        optimizer=optimizer,
        loss="categorical_crossentropy",
        metrics=[
            tf.keras.metrics.AUC(name='auc'),
            tf.keras.metrics.CategoricalAccuracy(name='acc')  # 使用普通准确率
        ]
    )
    
    # 回调函数
    callbacks_list = [
        callbacks.ModelCheckpoint(
            config.CHECKPOINT_PATH,
            monitor='val_auc',
            save_best_only=True,
            mode='max'
        ),
        callbacks.TensorBoard(config.LOG_DIR),
        UnfreezeCallback()
    ]
    
    # 训练
    history = model.fit(
        train_dataset,
        epochs=config.TRAIN_PARAMS["EPOCHS"],  # 必须指定epochs
        steps_per_epoch=steps_per_epoch,
        validation_data=val_dataset,
        validation_steps=validation_steps,
        callbacks=callbacks_list  # 确保回调列表已启用
    )
    return history

if __name__ == "__main__":
    train()