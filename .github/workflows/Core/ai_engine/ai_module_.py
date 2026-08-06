"""
Advanced, hyper-scalable AI model module for training and inference.

Features:
- Automatic distribution strategy (CPU / single-GPU / multi-GPU / multi-worker)
- Mixed precision and XLA support
- Configurable architectures: 'mlp', 'cnn', 'transformer'
- Efficient tf.data pipelines
- Modern callbacks and LR schedules
- Save / load as SavedModel
"""

import os
import math
from typing import Callable, Dict, Optional, Tuple, Any

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

# Optional: integrate with Weights & Biases if available
try:
    import wandb
    from wandb.keras import WandbCallback
    _WANDB_AVAILABLE = True
except Exception:
    _WANDB_AVAILABLE = False

AUTOTUNE = tf.data.AUTOTUNE


class AdvancedAIModel:
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        config: dictionary of options. Defaults provided below.
        Key options:
          - 'seed' : random seed
          - 'mixed_precision': True/False
          - 'enable_xla': True/False
          - 'strategy': None|'mirrored'|'multiworker' -- auto-detected if None
          - 'architecture': 'mlp'|'cnn'|'transformer'
          - 'input_shape': tuple
          - 'num_classes': int
          - 'dropout': float
          - 'hidden_units': list[int] for mlp
          - 'cnn_filters': list[int] for cnn
          - 'transformer' keys: num_heads, ff_dim, num_layers
          - 'optimizer': tf.keras optimizer instance or name
          - 'loss': loss instance or name
          - 'metrics': list of metric instances or names
        """
        self.default_config = {
            "seed": 42,
            "mixed_precision": True,
            "enable_xla": True,
            "strategy": None,
            "architecture": "mlp",
            "input_shape": (784,),
            "num_classes": 10,
            "dropout": 0.2,
            "hidden_units": [1024, 512, 256],
            "cnn_filters": [32, 64, 128],
            "transformer": {"num_heads": 4, "ff_dim": 256, "num_layers": 2},
            "optimizer": "adam",
            "loss": "sparse_categorical_crossentropy",
            "metrics": ["accuracy"],
            "seed_everything": True,
        }
        self.config = dict(self.default_config)
        if config:
            self.config.update(config)

        # Performance toggles
        if self.config["enable_xla"]:
            tf.config.optimizer.set_jit(True)
        if self.config["mixed_precision"]:
            try:
                from tensorflow.keras import mixed_precision
                mixed_precision.set_global_policy("mixed_float16")
            except Exception:
                # graceful fallback if TF build doesn't support mixed precision
                pass

        if self.config.get("seed_everything", True):
            self._set_seed(self.config["seed"])

        # pick distribution strategy
        self.strategy = self._get_strategy(self.config.get("strategy"))
        self.model: Optional[tf.keras.Model] = None
        self.is_compiled = False

    # -------------------------
    # Utilities
    # -------------------------
    @staticmethod
    def _set_seed(seed: int):
        tf.random.set_seed(seed)
        try:
            import numpy as np
            np.random.seed(seed)
        except Exception:
            pass
        try:
            import random
            random.seed(seed)
        except Exception:
            pass

    @staticmethod
    def _get_strategy(preferred: Optional[str]) -> tf.distribute.Strategy:
        """
        Decide and return an appropriate tf.distribute.Strategy.
        - If TF_CONFIG is set, MultiWorkerMirroredStrategy will be used.
        - If multiple GPUs are present, MirroredStrategy is used.
        - Else: DefaultStrategy (no distribution).
        """
        tf_config = os.environ.get("TF_CONFIG")
        gpus = tf.config.list_physical_devices("GPU")
        if tf_config:
            try:
                return tf.distribute.MultiWorkerMirroredStrategy()
            except Exception:
                pass
        if preferred == "mirrored" or (preferred is None and len(gpus) > 1):
            try:
                return tf.distribute.MirroredStrategy()
            except Exception:
                pass
        if preferred == "multiworker":
            try:
                return tf.distribute.MultiWorkerMirroredStrategy()
            except Exception:
                pass
        # fallback: one-device / default - safe for CPU or single GPU
        return tf.distribute.get_strategy()

    # -------------------------
    # Architectures
    # -------------------------
    def _build_mlp(self, input_shape: Tuple[int, ...]) -> tf.keras.Model:
        inp = keras.Input(shape=input_shape, name="inputs")
        x = inp
        for units in self.config.get("hidden_units", [1024, 512, 256]):
            x = layers.Dense(units, activation="relu")(x)
            x = layers.Dropout(self.config.get("dropout", 0.2))(x)
            x = layers.BatchNormalization()(x)
        out = layers.Dense(self.config["num_classes"], activation="softmax", dtype="float32")(x)
        return keras.Model(inputs=inp, outputs=out, name="mlp_model")

    def _build_cnn(self, input_shape: Tuple[int, ...]) -> tf.keras.Model:
        # expects image-like input: (H, W, C)
        inp = keras.Input(shape=input_shape, name="image_input")
        x = inp
        for filters in self.config.get("cnn_filters", [32, 64, 128]):
            x = layers.Conv2D(filters, 3, padding="same", activation="relu")(x)
            x = layers.BatchNormalization()(x)
            x = layers.MaxPool2D()(x)
        x = layers.Flatten()(x)
        x = layers.Dense(512, activation="relu")(x)
        x = layers.Dropout(self.config.get("dropout", 0.2))(x)
        out = layers.Dense(self.config["num_classes"], activation="softmax", dtype="float32")(x)
        return keras.Model(inputs=inp, outputs=out, name="cnn_model")

    def _build_transformer(self, input_shape: Tuple[int, ...]) -> tf.keras.Model:
        # Lightweight transformer for sequence tasks (expects (seq_len, d_model) or integer tokenized input)
        seq_len = input_shape[0]
        d_model = input_shape[1] if len(input_shape) > 1 else 128

        inp = keras.Input(shape=(seq_len,), dtype="int32", name="token_inputs")
        vocab_size = self.config.get("vocab_size", 30000)
        embedding = layers.Embedding(vocab_size, d_model)(inp)
        x = embedding
        tconf = self.config.get("transformer", {})
        for _ in range(tconf.get("num_layers", 2)):
            # multi-head attention block
            attn = layers.MultiHeadAttention(num_heads=tconf.get("num_heads", 4), key_dim=d_model)(x, x)
            x = layers.Add()([x, attn])
            x = layers.LayerNormalization()(x)
            ff = layers.Dense(tconf.get("ff_dim", 256), activation="relu")(x)
            ff = layers.Dense(d_model)(ff)
            x = layers.Add()([x, ff])
            x = layers.LayerNormalization()(x)
        x = layers.GlobalAveragePooling1D()(x)
        out = layers.Dense(self.config["num_classes"], activation="softmax", dtype="float32")(x)
        return keras.Model(inputs=inp, outputs=out, name="transformer_model")

    # -------------------------
    # Public API
    # -------------------------
    def build_model(self):
        """
        Build the model according to config. Should be called before compile/train.
        """
        input_shape = tuple(self.config["input_shape"])
        arch = self.config.get("architecture", "mlp").lower()

        with self.strategy.scope():
            if arch == "mlp":
                self.model = self._build_mlp(input_shape)
            elif arch == "cnn":
                self.model = self._build_cnn(input_shape)
            elif arch == "transformer":
                self.model = self._build_transformer(input_shape)
            else:
                raise ValueError(f"Unsupported architecture: {arch}")

        return self.model

    def compile_model(self, optimizer=None, loss=None, metrics=None, lr: Optional[float] = None):
        """
        Compile the keras model. Accepts optimizer or name and loss or name.
        """
        with self.strategy.scope():
            opt = optimizer or self.config.get("optimizer", "adam")
            if isinstance(opt, str):
                if lr is not None:
                    if opt.lower() == "adam":
                        opt = keras.optimizers.Adam(learning_rate=lr)
                    else:
                        opt = keras.optimizers.get(opt)
                else:
                    opt = keras.optimizers.get(opt)
            loss = loss or self.config.get("loss", "sparse_categorical_crossentropy")
            metrics = metrics or self.config.get("metrics", ["accuracy"])
            # If mixed precision is on, wrap optimizer for loss scaling automatically (TF does it)
            self.model.compile(optimizer=opt, loss=loss, metrics=metrics)
            self.is_compiled = True

    # -------------------------
    # Data helpers
    # -------------------------
    @staticmethod
    def _standardize_dataset(ds: tf.data.Dataset, batch_size: int, training: bool = True, cache: bool = True):
        if cache:
            ds = ds.cache()
        if training:
            ds = ds.shuffle(16 * batch_size)
        ds = ds.batch(batch_size, drop_remainder=False)
        ds = ds.prefetch(AUTOTUNE)
        return ds

    def dataset_from_numpy(self, x, y=None, batch_size: int = 64, training: bool = True, preprocess_fn: Optional[Callable] = None):
        """
        Build a tf.data.Dataset from numpy arrays. preprocess_fn maps (x, y) -> (x, y).
        """
        if y is None:
            ds = tf.data.Dataset.from_tensor_slices(x)
        else:
            ds = tf.data.Dataset.from_tensor_slices((x, y))
        if preprocess_fn:
            ds = ds.map(preprocess_fn, num_parallel_calls=AUTOTUNE)
        return self._standardize_dataset(ds, batch_size, training=training)

    # -------------------------
    # Training loop
    # -------------------------
    def _make_callbacks(self, output_dir: str, use_wandb: bool = False):
        callbacks = []
        os.makedirs(output_dir, exist_ok=True)
        ckpt = keras.callbacks.ModelCheckpoint(
            filepath=os.path.join(output_dir, "checkpoint-{epoch:02d}-{val_loss:.4f}.h5"),
            save_best_only=True,
            monitor="val_loss",
            verbose=1,
        )
        tb = keras.callbacks.TensorBoard(log_dir=os.path.join(output_dir, "logs"))
        rlr = keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=3, verbose=1)
        es = keras.callbacks.EarlyStopping(monitor="val_loss", patience=10, restore_best_weights=True, verbose=1)

        callbacks.extend([ckpt, tb, rlr, es])
        if use_wandb and _WANDB_AVAILABLE:
            callbacks.append(WandbCallback(save_model=False))
        return callbacks

    def train(
        self,
        train_ds: tf.data.Dataset,
        val_ds: Optional[tf.data.Dataset] = None,
        epochs: int = 10,
        steps_per_epoch: Optional[int] = None,
        validation_steps: Optional[int] = None,
        output_dir: str = "runs",
        use_wandb: bool = False,
    ):
        """
        Train using model.fit inside the chosen distribution strategy.
        """
        if self.model is None:
            self.build_model()
        if not self.is_compiled:
            # default compile with cosine decay schedule
            initial_lr = 1e-3
            total_steps = epochs * (steps_per_epoch or 100)
            lr_schedule = tf.keras.optimizers.schedules.CosineDecay(initial_learning_rate=initial_lr, decay_steps=total_steps)
            self.compile_model(optimizer=keras.optimizers.Adam(learning_rate=lr_schedule))

        callbacks = self._make_callbacks(output_dir, use_wandb=use_wandb)

        history = self.model.fit(
            train_ds,
            epochs=epochs,
            steps_per_epoch=steps_per_epoch,
            validation_data=val_ds,
            validation_steps=validation_steps,
            callbacks=callbacks,
        )
        return history

    # -------------------------
    # Evaluation / inference / save/load
    # -------------------------
    def evaluate(self, ds: tf.data.Dataset, steps: Optional[int] = None):
        if self.model is None:
            raise RuntimeError("Model not built/loaded.")
        return self.model.evaluate(ds, steps=steps)

    def predict(self, ds: tf.data.Dataset):
        if self.model is None:
            raise RuntimeError("Model not built/loaded.")
        return self.model.predict(ds)

    def save(self, path: str):
        """
        Save the model in SavedModel format for serving.
        """
        if self.model is None:
            raise RuntimeError("Model not built.")
        self.model.save(path, include_optimizer=True, save_format="tf")

    def load(self, path: str):
        """
        Load a SavedModel from disk.
        """
        with self.strategy.scope():
            self.model = tf.keras.models.load_model(path)
            self.is_compiled = True

    # -------------------------
    # Utility: quick example runner
    # -------------------------
    @staticmethod
    def example_mnist_run(output_dir: str = "runs/example", epochs: int = 3, batch_size: int = 128):
        """
        Quick runnable example: loads MNIST, builds the default MLP and trains briefly.
        Useful as a smoke test and local benchmark.
        """
        (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
        x_train = x_train.astype("float32") / 255.0
        x_test = x_test.astype("float32") / 255.0
        # flatten for mlp
        x_train = x_train.reshape((-1, 28 * 28))
        x_test = x_test.reshape((-1, 28 * 28))

        config = {"architecture": "mlp", "input_shape": (28 * 28,), "num_classes": 10}
        model_wrapper = AdvancedAIModel(config=config)
        model_wrapper.build_model()
        train_ds = model_wrapper.dataset_from_numpy(x_train, y_train, batch_size=batch_size, training=True)
        val_ds = model_wrapper.dataset_from_numpy(x_test, y_test, batch_size=batch_size, training=False)
        model_wrapper.train(train_ds, val_ds, epochs=epochs, output_dir=output_dir)
        print("Evaluation:", model_wrapper.evaluate(val_ds))

# End of file