import streamlit as st
import numpy as np
import tensorflow as tf
from PIL import Image

st.set_page_config(page_title="Waste Classification", layout="centered", page_icon="♻️")

# ── Custom Styles ────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Mono:wght@400;500&family=DM+Sans:wght@300;400;500;600&display=swap');

html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
}

.stApp {
    background-color: #F5F4F0;
}

#MainMenu, footer, header { visibility: hidden; }
.block-container {
    padding-top: 3rem;
    padding-bottom: 3rem;
    max-width: 820px;
}

/* ── Header ── */
.app-header {
    display: flex;
    align-items: center;
    gap: 14px;
    margin-bottom: 0.25rem;
}

.app-icon {
    width: 44px;
    height: 44px;
    background: #1C1C1C;
    border-radius: 12px;
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 22px;
    flex-shrink: 0;
}

.app-title {
    font-family: 'DM Sans', sans-serif;
    font-size: 1.6rem;
    font-weight: 600;
    color: #1C1C1C;
    letter-spacing: -0.02em;
    margin: 0;
    line-height: 1;
}

.app-subtitle {
    font-size: 0.875rem;
    color: #7A7A72;
    font-weight: 400;
    margin: 0.35rem 0 0 0;
}

.header-divider {
    height: 1px;
    background: #E0DFD8;
    margin: 1.5rem 0 2rem 0;
}

/* ── Upload Zone ── */
[data-testid="stFileUploader"] {
    background: #FFFFFF;
    border: 1.5px dashed #C8C7C0;
    border-radius: 14px;
    padding: 1.25rem;
    transition: border-color 0.2s;
}

[data-testid="stFileUploader"]:hover {
    border-color: #1C1C1C;
}

/* Fix invisible text inside upload box */
[data-testid="stFileUploaderDropzone"],
[data-testid="stFileUploaderDropzone"] p,
[data-testid="stFileUploaderDropzone"] span,
[data-testid="stFileUploaderDropzone"] div,
[data-testid="stFileUploaderDropzone"] small {
    background: transparent !important;
    color: #4A4A42 !important;
}

[data-testid="stFileUploader"] label,
[data-testid="stFileUploader"] span {
    color: #4A4A42 !important;
}

/* ── Result Card ── */
.result-card {
    background: #FFFFFF;
    border-radius: 14px;
    padding: 1.5rem 1.75rem 1.25rem 1.75rem;
    border: 1px solid #E8E7E0;
    box-shadow: 0 1px 3px rgba(0,0,0,0.04);
    margin-bottom: 0.75rem;
}

.prediction-chip {
    display: inline-block;
    background: #1C1C1C;
    color: #FFFFFF;
    font-family: 'DM Mono', monospace;
    font-size: 0.95rem;
    font-weight: 500;
    padding: 0.4rem 1rem;
    border-radius: 100px;
    letter-spacing: 0.01em;
}

.meta-label {
    font-size: 0.72rem;
    font-weight: 500;
    text-transform: uppercase;
    letter-spacing: 0.09em;
    color: #9A9A92;
    margin-top: 1.1rem;
    margin-bottom: 0.3rem;
}

.confidence-value {
    font-family: 'DM Mono', monospace;
    font-size: 2rem;
    font-weight: 500;
    color: #1C1C1C;
    line-height: 1;
}

/* ── Prob bar card ── */
.prob-card {
    background: #FFFFFF;
    border-radius: 14px;
    padding: 1.25rem 1.75rem;
    border: 1px solid #E8E7E0;
    box-shadow: 0 1px 3px rgba(0,0,0,0.04);
}

.prob-section-title {
    font-size: 0.72rem;
    font-weight: 500;
    text-transform: uppercase;
    letter-spacing: 0.09em;
    color: #9A9A92;
    margin-bottom: 1rem;
}

.prob-row {
    display: flex;
    align-items: center;
    gap: 10px;
    margin-bottom: 8px;
}

.prob-name {
    font-family: 'DM Mono', monospace;
    font-size: 0.78rem;
    color: #4A4A42;
    width: 78px;
    flex-shrink: 0;
}

.prob-name-bold {
    font-family: 'DM Mono', monospace;
    font-size: 0.78rem;
    font-weight: 600;
    color: #1C1C1C;
    width: 78px;
    flex-shrink: 0;
}

.prob-track {
    flex: 1;
    height: 6px;
    background: #EDECE8;
    border-radius: 100px;
    overflow: hidden;
}

.prob-fill-active {
    height: 6px;
    border-radius: 100px;
    background: #1C1C1C;
}

.prob-fill-inactive {
    height: 6px;
    border-radius: 100px;
    background: #C8C7C0;
}

.prob-pct {
    font-family: 'DM Mono', monospace;
    font-size: 0.78rem;
    color: #7A7A72;
    width: 38px;
    text-align: right;
    flex-shrink: 0;
}

/* ── Image ── */
[data-testid="stImage"] img {
    border-radius: 12px;
    border: 1px solid #E8E7E0;
}
</style>
""", unsafe_allow_html=True)

# ── Header ───────────────────────────────────────────────────────────────────
st.markdown("""
<div class="app-header">
    <div class="app-icon">♻️</div>
    <div>
        <p class="app-title">Waste Classification</p>
        <p class="app-subtitle">Material recognition powered by EfficientNet</p>
    </div>
</div>
<div class="header-divider"></div>
""", unsafe_allow_html=True)

# ── Model ─────────────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def load_model():
    interpreter = tf.lite.Interpreter(model_path="model.tflite")
    interpreter.allocate_tensors()
    return interpreter

interpreter = load_model()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

CLASS_NAMES = ['Cardboard', 'Glass', 'Metal', 'Paper', 'Plastic', 'Trash']

# ── Upload ────────────────────────────────────────────────────────────────────
uploaded_file = st.file_uploader(
    "Drop an image here, or click to browse",
    type=["jpg", "jpeg", "png"],
    label_visibility="collapsed"
)

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert('RGB')

    col1, col2 = st.columns([1, 1], gap="large")

    with col1:
        st.image(image, caption="", use_container_width=True)

    with col2:
        # ── Inference ──
        img_array = np.array(image.resize((224, 224)), dtype=np.float32)
        img_array = np.expand_dims(img_array, axis=0)
        img_array = tf.keras.applications.efficientnet.preprocess_input(img_array)

        interpreter.set_tensor(input_details[0]['index'], img_array)
        interpreter.invoke()
        output_data = interpreter.get_tensor(output_details[0]['index'])[0]

        predicted_index = int(np.argmax(output_data))
        confidence = float(output_data[predicted_index])
        predicted_label = CLASS_NAMES[predicted_index]

        # ── Top result card ──
        st.markdown(f"""
        <div class="result-card">
            <div class="meta-label">Detected Material</div>
            <span class="prediction-chip">{predicted_label}</span>
            <div class="meta-label">Confidence</div>
            <div class="confidence-value">{confidence:.1%}</div>
        </div>
        """, unsafe_allow_html=True)

        # ── Probability breakdown ──
        # Build rows as a plain string first, then inject once — avoids f-string/HTML conflicts
        rows_html = ""
        for i, prob in enumerate(output_data):
            p = float(prob)
            is_top = (i == predicted_index)
            name_class = "prob-name-bold" if is_top else "prob-name"
            fill_class = "prob-fill-active" if is_top else "prob-fill-inactive"
            pct_str = f"{p:.0%}"
            width_str = f"{p * 100:.1f}%"
            rows_html += (
                '<div class="prob-row">'
                f'<span class="{name_class}">{CLASS_NAMES[i]}</span>'
                '<div class="prob-track">'
                f'<div class="{fill_class}" style="width:{width_str}"></div>'
                '</div>'
                f'<span class="prob-pct">{pct_str}</span>'
                '</div>'
            )

        st.markdown(
            '<div class="prob-card">'
            '<div class="prob-section-title">Probability Breakdown</div>'
            + rows_html +
            '</div>',
            unsafe_allow_html=True
        )