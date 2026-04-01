import json
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
import tensorflow as tf
from PIL import Image

from src.preprocess_image import preprocess_pil_image
from src.train_model import CLASS_NAMES, train_and_evaluate

MODEL_DIR = "model"
MODEL_PATH = os.path.join(MODEL_DIR, "fashion_mnist.keras")
HISTORY_PATH = os.path.join(MODEL_DIR, "history.json")
CM_PATH = os.path.join(MODEL_DIR, "confusion_matrix.npy")
METRICS_PATH = os.path.join(MODEL_DIR, "metrics.json")
TEST_CSV_PATH = "data/fashion-mnist_test.csv"


@st.cache_resource
def load_model(model_path: str) -> tf.keras.Model:
    return tf.keras.models.load_model(model_path)


@st.cache_data
def load_history(history_path: str) -> dict:
    with open(history_path, "r", encoding="utf-8") as f:
        return json.load(f)


@st.cache_data
def load_metrics(metrics_path: str) -> dict:
    with open(metrics_path, "r", encoding="utf-8") as f:
        return json.load(f)


@st.cache_data
def load_confusion_matrix(cm_path: str) -> np.ndarray:
    return np.load(cm_path)


@st.cache_data
def load_test_dataset(test_csv_path: str) -> pd.DataFrame:
    return pd.read_csv(test_csv_path)


def draw_history_plots(history: dict) -> None:
    hist_df = pd.DataFrame(history)

    chart_df = pd.DataFrame(
        {
            "train_accuracy": hist_df.get("accuracy", pd.Series(dtype=float)),
            "val_accuracy": hist_df.get("val_accuracy", pd.Series(dtype=float)),
        }
    )
    st.subheader("Accuracy Curve")
    st.line_chart(chart_df)

    loss_df = pd.DataFrame(
        {
            "train_loss": hist_df.get("loss", pd.Series(dtype=float)),
            "val_loss": hist_df.get("val_loss", pd.Series(dtype=float)),
        }
    )
    st.subheader("Loss Curve")
    st.line_chart(loss_df)


def draw_confusion_matrix(cm: np.ndarray) -> None:
    fig, ax = plt.subplots(figsize=(8, 7))
    image = ax.imshow(cm, cmap="Blues")
    ax.set_title("Confusion Matrix")
    ax.set_xlabel("Predicted Label")
    ax.set_ylabel("True Label")
    ax.set_xticks(range(len(CLASS_NAMES)))
    ax.set_yticks(range(len(CLASS_NAMES)))
    ax.set_xticklabels(CLASS_NAMES, rotation=45, ha="right")
    ax.set_yticklabels(CLASS_NAMES)

    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center", color="black", fontsize=8)

    fig.colorbar(image, ax=ax)
    fig.tight_layout()
    st.pyplot(fig)


def ensure_model_artifacts() -> bool:
    return (
        os.path.exists(MODEL_PATH)
        and os.path.exists(HISTORY_PATH)
        and os.path.exists(METRICS_PATH)
        and os.path.exists(CM_PATH)
    )


def predict_image(model: tf.keras.Model, image: Image.Image, invert_mode: str) -> tuple:
    processed_input, processed_2d = preprocess_pil_image(image, invert_mode=invert_mode)
    predictions = model.predict(processed_input, verbose=0)[0]
    pred_idx = int(np.argmax(predictions))
    confidence = float(predictions[pred_idx])
    return pred_idx, confidence, predictions, processed_2d


def predict_array(model: tf.keras.Model, image_2d: np.ndarray) -> tuple:
    model_input = image_2d.astype("float32").reshape(1, 28, 28, 1) / 255.0
    predictions = model.predict(model_input, verbose=0)[0]
    pred_idx = int(np.argmax(predictions))
    confidence = float(predictions[pred_idx])
    return pred_idx, confidence, predictions


def main() -> None:
    st.set_page_config(page_title="Fashion Classifier", layout="wide")
    st.title("Fashion-MNIST Classifier Dashboard")
    st.caption("Use dataset samples for a reliable demo, then optionally try uploaded images.")
    st.info(
        "Demo note: this model was trained on Fashion-MNIST. It performs best on dataset-style images "
        "and may fail on real photos from outside the dataset."
    )

    with st.sidebar:
        st.header("Model Management")
        if st.button("Train / Retrain Model", use_container_width=True):
            with st.spinner("Training model and generating artifacts..."):
                result = train_and_evaluate(output_dir=MODEL_DIR)
            st.success("Training complete")
            st.json(result)
            st.cache_resource.clear()
            st.cache_data.clear()

    if not ensure_model_artifacts():
        st.warning("Model artifacts are missing. Click 'Train / Retrain Model' in the sidebar.")
        st.stop()

    model = load_model(MODEL_PATH)
    history = load_history(HISTORY_PATH)
    metrics = load_metrics(METRICS_PATH)
    cm = load_confusion_matrix(CM_PATH)
    test_df = load_test_dataset(TEST_CSV_PATH)

    if "label" not in test_df.columns:
        st.error("Test dataset is missing the 'label' column. Random dataset demo is unavailable.")
        st.stop()

    col1, col2 = st.columns([1, 1])

    with col1:
        st.subheader("Image Prediction")

        st.markdown("### Demo with random dataset image")
        if "demo_sample_idx" not in st.session_state:
            st.session_state.demo_sample_idx = int(np.random.randint(0, len(test_df)))

        if st.button("Pick random image from Fashion-MNIST test set", use_container_width=True):
            st.session_state.demo_sample_idx = int(np.random.randint(0, len(test_df)))

        sample_row = test_df.iloc[st.session_state.demo_sample_idx]
        true_label = int(sample_row["label"])
        sample_pixels = sample_row.drop(labels=["label"]).to_numpy(dtype=np.uint8).reshape(28, 28)
        sample_image = Image.fromarray(sample_pixels, mode="L")

        sample_pred_idx, sample_confidence, sample_predictions = predict_array(model, sample_pixels)

        dataset_col1, dataset_col2 = st.columns(2)
        with dataset_col1:
            st.markdown("Dataset image (28x28)")
            st.image(sample_image, clamp=True, use_container_width=True)
        with dataset_col2:
            st.markdown("Ground truth vs prediction")
            st.write(f"True label: **{CLASS_NAMES[true_label]}**")
            st.write(f"Predicted: **{CLASS_NAMES[sample_pred_idx]}**")
            st.write(f"Confidence: **{sample_confidence:.2%}**")

        sample_probs = pd.DataFrame(
            {"Class": CLASS_NAMES, "Probability": sample_predictions}
        ).sort_values("Probability", ascending=False)
        st.dataframe(sample_probs, use_container_width=True)

        st.divider()
        st.markdown("### Optional: upload external image")
        invert_mode = st.selectbox(
            "Preprocessing mode",
            options=["auto", "always", "never"],
            index=0,
            help="Use 'auto' in most cases. If predictions look wrong, try 'always' or 'never'.",
        )
        uploaded_file = st.file_uploader("Upload an image", type=["png", "jpg", "jpeg", "bmp", "webp"])

        if uploaded_file is not None:
            image = Image.open(uploaded_file)
            pred_idx, confidence, predictions, processed = predict_image(
                model, image, invert_mode=invert_mode
            )

            preview_col1, preview_col2 = st.columns(2)
            with preview_col1:
                st.markdown("Original Image")
                st.image(image, use_container_width=True)
            with preview_col2:
                st.markdown("Processed 28x28 Grayscale")
                st.image(processed, clamp=True, use_container_width=True)

            st.success(f"Prediction: {CLASS_NAMES[pred_idx]} ({confidence:.2%})")
            probs = pd.DataFrame(
                {"Class": CLASS_NAMES, "Probability": predictions}
            ).sort_values("Probability", ascending=False)
            st.dataframe(probs, use_container_width=True)

    with col2:
        st.subheader("Evaluation Summary")
        st.metric("Test Accuracy", f"{metrics.get('test_accuracy', 0.0):.2%}")
        st.metric("Test Loss", f"{metrics.get('test_loss', 0.0):.4f}")
        st.metric("Epochs Used", f"{metrics.get('epochs', 0)}")

    st.divider()
    draw_history_plots(history)

    st.divider()
    st.subheader("Confusion Matrix on Test Set")
    draw_confusion_matrix(cm)


if __name__ == "__main__":
    main()
