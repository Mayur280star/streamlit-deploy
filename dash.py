import streamlit as st
import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
from io import StringIO

# ======================================================
# CONFIG
# ======================================================
SEQUENCE_LENGTH = 2000
DEFAULT_THRESHOLD = 0.5
MODEL_PATH = "F:\\newworldbegins\\exonew\\Streamlit\\exoplanet_model.keras"  # Update this path as needed

st.set_page_config(
    page_title="EXO-SCAN AI",
    page_icon="🪐",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ======================================================
# CUSTOM ATTENTION LAYER
# ======================================================
class AttentionLayer(tf.keras.layers.Layer):
    def __init__(self, units=128, **kwargs):
        super().__init__(**kwargs)
        self.units = units

    def build(self, input_shape):
        self.W = self.add_weight(
            shape=(input_shape[-1], self.units),
            initializer="glorot_uniform",
            name="attention_W",
        )
        self.b = self.add_weight(
            shape=(self.units,), initializer="zeros", name="attention_b"
        )
        self.u = self.add_weight(
            shape=(self.units,), initializer="glorot_uniform", name="attention_u"
        )

    def call(self, x):
        uit = tf.tanh(tf.tensordot(x, self.W, axes=1) + self.b)
        ait = tf.tensordot(uit, self.u, axes=1)
        ait = tf.nn.softmax(ait, axis=1)
        ait = tf.expand_dims(ait, -1)
        return tf.reduce_sum(x * ait, axis=1)

    def get_config(self):
        config = super().get_config()
        config.update({"units": self.units})
        return config


# ======================================================
# LOAD MODEL
# ======================================================
@st.cache_resource
def load_model():
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Model not found at {MODEL_PATH}")
    return tf.keras.models.load_model(
        MODEL_PATH, custom_objects={"AttentionLayer": AttentionLayer}
    )


model = load_model()


# ======================================================
# HELPERS
# ======================================================
def preprocess_flux(flux):
    """Preprocess flux: normalize to mean 0, std 1 (assuming this matches training preprocessing).
    Adjust based on your dataset's preprocessing if different (e.g., divide by median for relative flux).
    This should fix the always-100% issue if inputs weren't normalized.
    """
    if np.std(flux) == 0:
        return flux  # Avoid division by zero
    flux = (flux - np.mean(flux)) / np.std(flux)
    # Alternative for light curves: flux = flux / np.median(flux)  # Uncomment if fluxes are around 1
    return flux


def fix_length(arr, target):
    if len(arr) > target:
        return arr[:target]
    return np.pad(arr, (0, target - len(arr)), mode="constant", constant_values=0)


def get_attention_weights(model, X):
    """Extract attention weights from the model for visualization."""
    # Find the attention layer
    attention_layer = None
    for layer in model.layers:
        if isinstance(layer, AttentionLayer):
            attention_layer = layer
            break
    if attention_layer is None:
        return None

    # Create a submodel up to the LSTM before attention
    lstm_layer_name = "bidirectional_1"  # Adjust based on your model summary (lstm2)
    submodel = tf.keras.Model(
        inputs=model.input, outputs=model.get_layer(lstm_layer_name).output
    )
    lstm_output = submodel.predict(X, verbose=0)

    # Get attention scores
    uit = tf.tanh(
        tf.tensordot(lstm_output, attention_layer.W, axes=1) + attention_layer.b
    )
    ait = tf.tensordot(uit, attention_layer.u, axes=1)
    ait = tf.nn.softmax(ait, axis=1)
    return ait[0].numpy().flatten()  # Flatten for plotting


# ======================================================
# SIDEBAR CONFIGURATIONS
# ======================================================
with st.sidebar:
    st.header("⚙️ Configurations")
    threshold = st.slider(
        "Detection Threshold",
        min_value=0.0,
        max_value=1.0,
        value=DEFAULT_THRESHOLD,
        step=0.01,
    )
    normalize_method = st.selectbox(
        "Normalization Method",
        ["Z-Score (Mean=0, Std=1)", "Median Normalization", "None"],
    )
    show_attention = st.checkbox("Show Attention Visualization", value=True)
    show_model_summary = st.checkbox("Show Model Summary", value=False)

    st.divider()
    st.header("📊 Batch Results")
    download_results = st.button("Download All Results as CSV")

# ======================================================
# ABOUT SECTION & MODEL INFO
# ======================================================
st.title("🛸 EXO-SCAN AI")
st.caption(
    "Advanced Hybrid CNN-BiLSTM with Attention Mechanism for NASA-Grade Exoplanet Detection"
)

with st.expander("ℹ️ About the Model", expanded=True):
    st.markdown(
        """
    **Model Architecture Overview:**
    - **Input:** Light curve flux sequences (2000 timesteps).
    - **CNN Blocks:** Multi-scale convolutional layers (kernels: 5, 11, 21) with 64-256 filters for feature extraction, batch norm, ReLU, max pooling, and dropout (0.3).
    - **BiLSTM Layers:** Bidirectional LSTMs (128 & 64 units) to capture temporal dependencies in transit signals.
    - **Attention Mechanism:** Self-attention layer (128 units) to focus on potential transit events.
    - **Dense Layers:** Fully connected layers (256 & 128 units) with ReLU, batch norm, and dropout for classification.
    - **Output:** Sigmoid activation for binary classification (Exoplanet: 1, No Exoplanet: 0).
    - **Training:** Focal loss for imbalance, Adam optimizer (LR=0.0001), class weights, early stopping, and L2 regularization (0.0001).
    - **Metrics Focus:** Optimized for high recall to minimize false negatives in rare exoplanet detections.
    - **Total Parameters:** ~1.5M (based on summary).
    
    This model is trained on preprocessed Kepler/TESS-like datasets with positive/negative splits.
    For best results, ensure uploaded CSVs have 'flux' column with similar preprocessing.
    """
    )

    if show_model_summary:
        summary_io = StringIO()
        model.summary(print_fn=lambda x: summary_io.write(x + "\n"))
        st.code(summary_io.getvalue())

st.divider()

# ======================================================
# UPLOAD & PROCESSING
# ======================================================
files = st.file_uploader(
    "📡 Upload Light-Curve CSV Files (time, flux)",
    type=["csv"],
    accept_multiple_files=True,
)

results = []  # For batch download

if files:
    st.header("🔍 Scan Results")
    for file in files:
        with st.expander(f"🪐 Target: {file.name}", expanded=True):
            try:
                df = pd.read_csv(file)
                if "flux" not in df.columns:
                    st.error("❌ CSV must contain a 'flux' column.")
                    continue

                flux_raw = df["flux"].values
                if normalize_method == "Z-Score (Mean=0, Std=1)":
                    flux = preprocess_flux(flux_raw)
                elif normalize_method == "Median Normalization":
                    median = np.median(flux_raw)
                    flux = flux_raw / median if median != 0 else flux_raw
                else:
                    flux = flux_raw

                flux = fix_length(flux, SEQUENCE_LENGTH)
                X = flux.reshape(1, SEQUENCE_LENGTH, 1)

                # Prediction
                prob = float(model.predict(X, verbose=0)[0][0])
                confidence = prob * 100
                is_exoplanet = prob >= threshold

                results.append(
                    {
                        "File": file.name,
                        "Probability": prob,
                        "Confidence (%)": confidence,
                        "Detected": "Yes" if is_exoplanet else "No",
                        "Threshold": threshold,
                    }
                )

                col1, col2 = st.columns([3, 1])

                with col1:
                    # Light Curve Plot
                    fig, ax = plt.subplots(figsize=(12, 4))
                    ax.plot(
                        flux, color="#00e5ff", linewidth=1.2, label="Normalized Flux"
                    )
                    ax.set_facecolor("#050816")
                    ax.set_title("Light Curve Signal", color="white", fontsize=16)
                    ax.tick_params(colors="white")
                    ax.legend()
                    st.pyplot(fig)

                    # Attention Visualization if enabled
                    if show_attention:
                        attention_weights = get_attention_weights(model, X)
                        if attention_weights is not None:
                            # Attention is on downsampled sequence (after pooling), approximate scaling
                            scale_factor = SEQUENCE_LENGTH // len(attention_weights)
                            attention_resized = np.repeat(
                                attention_weights, scale_factor
                            )[:SEQUENCE_LENGTH]

                            fig_att, ax_att = plt.subplots(figsize=(12, 2))
                            sns.heatmap(
                                [attention_resized],
                                cmap="viridis",
                                cbar=True,
                                ax=ax_att,
                            )
                            ax_att.set_title(
                                "Attention Weights (Focus on Transits)",
                                color="white",
                                fontsize=16,
                            )
                            ax_att.set_xticks([])
                            ax_att.set_yticks([])
                            st.pyplot(fig_att)

                with col2:
                    st.markdown("### 🧠 AI Verdict")
                    if is_exoplanet:
                        st.success("🌍 EXOPLANET DETECTED")
                    else:
                        st.warning("☄️ NO TRANSIT SIGNAL")
                    st.metric("Detection Confidence", f"{confidence:.2f}%")
                    st.progress(prob)
                    st.metric("Raw Probability", f"{prob:.4f}")

            except Exception as e:
                st.error(f"Error processing {file.name}: {str(e)}")

    # Batch Download
    if download_results and results:
        results_df = pd.DataFrame(results)
        csv = results_df.to_csv(index=False)
        st.download_button(
            label="📥 Download Results CSV",
            data=csv,
            file_name="exo_scan_results.csv",
            mime="text/csv",
        )

else:
    st.info(
        "👆 Upload Kepler/TESS light-curve CSVs to begin scan. Multiple files supported for batch processing."
    )

# ======================================================
# FOOTER
# ======================================================
st.divider()
st.caption(
    "EXO-SCAN AI • Deep Learning for Space Discovery • Optimized for Researchers • Contact: [Your Email/LinkedIn]"
)
st.markdown(
    """
<style>
    body { background-color: #0a192f; color: white; }
    .stApp { background-color: #0a192f; }
    section[data-testid="stSidebar"] { background-color: #112240; }
    .css-1aumxhk { background-color: #112240; }
</style>
""",
    unsafe_allow_html=True,
)
