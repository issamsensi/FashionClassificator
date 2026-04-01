import argparse
import json
import os
import random

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split

tf.keras.utils.set_random_seed(42)
np.random.seed(42)
random.seed(42)

CLASS_NAMES = [
    "T-shirt/top",
    "Trouser",
    "Pullover",
    "Dress",
    "Coat",
    "Sandal",
    "Shirt",
    "Sneaker",
    "Bag",
    "Ankle boot",
]


def build_model() -> tf.keras.Model:
    model = tf.keras.Sequential(
        [
            tf.keras.layers.Input(shape=(28, 28, 1)),
            tf.keras.layers.Conv2D(32, 3, padding="same", activation="relu"),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Conv2D(64, 3, padding="same", activation="relu"),
            tf.keras.layers.MaxPooling2D(),
            tf.keras.layers.Dropout(0.25),
            tf.keras.layers.Conv2D(128, 3, padding="same", activation="relu"),
            tf.keras.layers.MaxPooling2D(),
            tf.keras.layers.Dropout(0.30),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(256, activation="relu"),
            tf.keras.layers.Dropout(0.40),
            tf.keras.layers.Dense(10, activation="softmax"),
        ]
    )
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def _ensure_output_dir(output_dir: str) -> None:
    os.makedirs(output_dir, exist_ok=True)


def train_and_evaluate(
    train_csv: str = "data/fashion-mnist_train.csv",
    test_csv: str = "data/fashion-mnist_test.csv",
    output_dir: str = "model",
    epochs: int = 20,
    batch_size: int = 256,
) -> dict:
    _ensure_output_dir(output_dir)

    train_df = pd.read_csv(train_csv)
    test_df = pd.read_csv(test_csv)

    if "label" not in train_df.columns:
        raise ValueError("Training CSV must contain a 'label' column.")

    X = train_df.drop(columns=["label"]).to_numpy(dtype="float32") / 255.0
    y = train_df["label"].to_numpy(dtype="int64")
    X = X.reshape(-1, 28, 28, 1)

    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.15, random_state=42, stratify=y
    )

    test_has_label = "label" in test_df.columns
    X_test = test_df.drop(columns=["label"], errors="ignore").to_numpy(dtype="float32") / 255.0
    X_test = X_test.reshape(-1, 28, 28, 1)
    y_test_true = test_df["label"].to_numpy(dtype="int64") if test_has_label else None

    autotune = tf.data.AUTOTUNE
    train_ds = (
        tf.data.Dataset.from_tensor_slices((X_train, y_train))
        .shuffle(len(X_train), seed=42)
        .batch(batch_size)
        .prefetch(autotune)
    )
    val_ds = tf.data.Dataset.from_tensor_slices((X_val, y_val)).batch(batch_size).prefetch(autotune)
    test_ds = tf.data.Dataset.from_tensor_slices(X_test).batch(batch_size).prefetch(autotune)

    model = build_model()

    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_accuracy", patience=5, restore_best_weights=True
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss", factor=0.5, patience=2, min_lr=1e-5
        ),
        tf.keras.callbacks.ModelCheckpoint(
            filepath=os.path.join(output_dir, "best_fashion_mnist.keras"),
            monitor="val_accuracy",
            save_best_only=True,
        ),
    ]

    history = model.fit(
        train_ds, validation_data=val_ds, epochs=epochs, callbacks=callbacks, verbose=1
    )

    model_path = os.path.join(output_dir, "fashion_mnist.keras")
    model.save(model_path)

    metrics = {
        "epochs": len(history.history.get("loss", [])),
        "train_samples": int(X_train.shape[0]),
        "val_samples": int(X_val.shape[0]),
        "test_samples": int(X_test.shape[0]),
    }

    if test_has_label:
        test_eval = model.evaluate(X_test, y_test_true, verbose=0)
        y_pred = np.argmax(model.predict(test_ds, verbose=0), axis=1)
        cm = confusion_matrix(y_test_true, y_pred)
        report = classification_report(
            y_test_true,
            y_pred,
            target_names=CLASS_NAMES,
            output_dict=True,
            zero_division=0,
        )

        metrics["test_loss"] = float(test_eval[0])
        metrics["test_accuracy"] = float(test_eval[1])

        np.save(os.path.join(output_dir, "confusion_matrix.npy"), cm)
        with open(os.path.join(output_dir, "classification_report.json"), "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2)

    with open(os.path.join(output_dir, "history.json"), "w", encoding="utf-8") as f:
        json.dump(history.history, f, indent=2)

    with open(os.path.join(output_dir, "metrics.json"), "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    return {
        "model_path": model_path,
        "metrics": metrics,
        "history_keys": list(history.history.keys()),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Train Fashion-MNIST model and save artifacts.")
    parser.add_argument("--train_csv", default="data/fashion-mnist_train.csv")
    parser.add_argument("--test_csv", default="data/fashion-mnist_test.csv")
    parser.add_argument("--output_dir", default="model")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=256)
    args = parser.parse_args()

    result = train_and_evaluate(
        train_csv=args.train_csv,
        test_csv=args.test_csv,
        output_dir=args.output_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()