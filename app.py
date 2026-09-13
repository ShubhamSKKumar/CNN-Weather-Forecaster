import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

# -----------------------------
# Page configuration
# -----------------------------
st.set_page_config(
    page_title="Weather Forecasting AI",
    page_icon="🌦️",
    layout="centered"
)

# -----------------------------
# Custom styling
# -----------------------------
st.markdown("""
<style>
    .main-title {
        text-align: center;
        font-size: 42px;
        font-weight: 700;
        margin-bottom: 5px;
    }

    .subtitle {
        text-align: center;
        font-size: 18px;
        color: #666;
        margin-bottom: 30px;
    }

    .result-box {
        padding: 25px;
        border-radius: 15px;
        text-align: center;
        border: 1px solid #ddd;
        margin-top: 20px;
    }

    .prediction {
        font-size: 36px;
        font-weight: 700;
        margin: 10px 0;
    }

    .confidence {
        font-size: 20px;
    }
</style>
""", unsafe_allow_html=True)

# -----------------------------
# Title
# -----------------------------
st.markdown(
    '<div class="main-title">🌦️ Weather Forecasting AI</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">'
    'Upload an image of the sky and let the ResNet50 model predict the weather.'
    '</div>',
    unsafe_allow_html=True
)

# -----------------------------
# Model loading
# -----------------------------
@st.cache_resource
def load_model():
    return tf.keras.models.load_model(
        "models/02_resnet50_weather_classifier.keras",
        compile=False
    )

model = load_model()

classes = ["Cloudy", "Rain", "Shine", "Sunrise"]

# -----------------------------
# Sidebar
# -----------------------------
with st.sidebar:
    st.header("🤖 Model Information")

    st.write("**Model:** ResNet50")
    st.write("**Task:** Weather Classification")
    st.write("**Input Size:** 224 × 224")
    st.write("**Classes:**")

    for weather_class in classes:
        st.write(f"• {weather_class}")

# -----------------------------
# Image uploader
# -----------------------------
uploaded_file = st.file_uploader(
    "📷 Choose a sky image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:

    # Open image
    image = Image.open(uploaded_file).convert("RGB")

    st.image(
        image,
        caption="Uploaded Image",
        use_container_width=True
    )

    # -----------------------------
    # Prediction
    # -----------------------------
    with st.spinner("🤖 Analyzing the image..."):

        img = image.resize((224, 224))
        img_array = tf.keras.preprocessing.image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)

        predictions = model.predict(img_array, verbose=0)[0]

        predicted_index = np.argmax(predictions)
        predicted_class = classes[predicted_index]
        confidence = predictions[predicted_index] * 100

    # -----------------------------
    # Result
    # -----------------------------
    st.markdown(
        f"""
        <div class="result-box">
            <div>Predicted Weather</div>
            <div class="prediction">{predicted_class}</div>
            <div class="confidence">
                Confidence: <b>{confidence:.2f}%</b>
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

    # -----------------------------
    # Confidence warning
    # -----------------------------
    if confidence < 60:
        st.warning(
            "⚠️ The model has relatively low confidence in this prediction."
        )
    elif confidence < 80:
        st.info(
            "ℹ️ The model has moderate confidence in this prediction."
        )
    else:
        st.success(
            "✅ The model is highly confident in this prediction."
        )

    # -----------------------------
    # All class probabilities
    # -----------------------------
    st.subheader("📊 Prediction Probabilities")

    for i, weather_class in enumerate(classes):
        probability = float(predictions[i])

        st.write(
            f"**{weather_class}** — {probability * 100:.2f}%"
        )

        st.progress(probability)

else:
    st.info("👆 Upload an image to get a weather prediction.")
