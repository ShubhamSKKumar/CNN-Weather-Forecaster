import json

import matplotlib.pyplot as plt
import numpy as np
import streamlit as st
import tensorflow as tf
from PIL import Image


# =========================================================
# PAGE CONFIGURATION
# =========================================================

st.set_page_config(
    page_title="Weather Forecasting AI",
    page_icon="🌦️",
    layout="centered"
)

# =========================================================
# LIGHT WEATHER-THEMED DESIGN
# =========================================================

st.markdown(
    """
    <style>
        .stApp {
            background: linear-gradient(
                180deg,
                #EAF6FF 0%,
                #FFFFFF 55%,
                #F5FBFF 100%
            );
        }
    </style>
    """,
    unsafe_allow_html=True
)
# =========================================================
# WEATHER CLASSES AND ICONS
# =========================================================

classes = ["Cloudy", "Rain", "Shine", "Sunrise"]

weather_icons = {
    "Cloudy": "☁️",
    "Rain": "🌧️",
    "Shine": "☀️",
    "Sunrise": "🌅"
}


# =========================================================
# TITLE
# =========================================================

st.title("🌦️ Weather Forecasting AI")

st.write(
    "Upload an image of the sky and let the ResNet50 AI "
    "predict the weather."
)


# =========================================================
# LOAD MODEL
# =========================================================

@st.cache_resource
def load_model():
    return tf.keras.models.load_model(
        "models/02_resnet50_weather_classifier.keras",
        compile=False
    )


model = load_model()


# =========================================================
# SIDEBAR
# =========================================================

with st.sidebar:

    st.header("🤖 About the Model")

    st.write("**Architecture:** ResNet50")
    st.write("**Task:** Weather Classification")
    st.write("**Input Size:** 224 × 224")
    st.write("**Number of Classes:** 4")

    st.subheader("Weather Classes")

    for weather_class in classes:
        icon = weather_icons[weather_class]
        st.write(f"{icon} {weather_class}")

    st.markdown("---")

    st.header("📈 Model Performance")

    st.metric(
        "Best Validation Accuracy",
        "97.33%"
    )

    st.metric(
        "Final Validation Accuracy",
        "96.89%"
    )

    st.metric(
        "Final Training Accuracy",
        "98.00%"
    )

    st.markdown("---")

    st.caption(
        "ResNet50 transfer learning with data augmentation."
    )


# =========================================================
# IMAGE UPLOADER
# =========================================================

uploaded_file = st.file_uploader(
    "📷 Choose a sky image",
    type=["jpg", "jpeg", "png"]
)


# =========================================================
# MAIN APPLICATION
# =========================================================

if uploaded_file is not None:

    # -----------------------------------------------------
    # DISPLAY UPLOADED IMAGE
    # -----------------------------------------------------

    image = Image.open(uploaded_file).convert("RGB")

    st.image(
        image,
        caption="Uploaded Image",
        use_container_width=True
    )


    # -----------------------------------------------------
    # PREPARE IMAGE
    # -----------------------------------------------------

    img = image.resize((224, 224))

    img_array = tf.keras.preprocessing.image.img_to_array(img)

    img_array = np.expand_dims(img_array, axis=0)

    # IMPORTANT:
    # Do NOT apply resnet50.preprocess_input() here.
    #
    # Your saved model already contains:
    #
    # data_augmentation
    #        ↓
    # ResNet50 preprocess_input
    #        ↓
    # ResNet50
    #
    # Therefore the raw image array is passed to model.predict().


    # -----------------------------------------------------
    # PREDICTION
    # -----------------------------------------------------

    with st.spinner("🤖 Analyzing the image..."):

        predictions = model.predict(
            img_array,
            verbose=0
        )[0]


    # -----------------------------------------------------
    # GET PREDICTION RESULT
    # -----------------------------------------------------

    predicted_index = int(np.argmax(predictions))

    predicted_class = classes[predicted_index]

    confidence = float(predictions[predicted_index]) * 100

    icon = weather_icons[predicted_class]


    # =====================================================
    # PREDICTION RESULT
    # =====================================================

    st.subheader("🌦️ Prediction")

    st.markdown(
        f"## {icon} {predicted_class}"
    )

    st.metric(
        "Confidence",
        f"{confidence:.2f}%"
    )


    # =====================================================
    # CONFIDENCE MESSAGE
    # =====================================================

    if confidence >= 80:

        st.success(
            "✅ The model is highly confident in this prediction."
        )

    elif confidence >= 60:

        st.info(
            "ℹ️ The model has moderate confidence in this prediction."
        )

    else:

        st.warning(
            "⚠️ The model has low confidence. "
            "Try uploading a clearer sky image."
        )


    # =====================================================
    # PREDICTION PROBABILITIES
    # =====================================================

    st.subheader("📊 Prediction Probabilities")

    # Sort classes from highest probability to lowest
    sorted_indices = np.argsort(predictions)[::-1]


    for index in sorted_indices:

        weather_class = classes[index]

        probability = float(predictions[index])

        percentage = probability * 100

        icon = weather_icons[weather_class]

        st.write(
            f"{icon} **{weather_class}** — "
            f"{percentage:.2f}%"
        )

        st.progress(
            min(max(probability, 0.0), 1.0)
        )


    # =====================================================
    # AI ANALYSIS
    # =====================================================

    st.subheader("🧠 AI Analysis")

    second_index = int(sorted_indices[1])

    second_class = classes[second_index]

    second_probability = (
        float(predictions[second_index]) * 100
    )

    difference = confidence - second_probability


    st.write(
        f"The model predicts **{predicted_class}** as the "
        f"most likely weather condition with "
        f"**{confidence:.2f}% confidence**."
    )

    st.write(
        f"The second most likely class is **{second_class}** "
        f"with **{second_probability:.2f}% probability**."
    )

    st.write(
        f"The difference between the top two predictions is "
        f"**{difference:.2f} percentage points**."
    )


    # =====================================================
    # TRAINING PERFORMANCE
    # =====================================================

    st.subheader("📈 Training Performance")


    try:

        with open("training_history.json", "r") as file:

            history_data = json.load(file)


        epochs = range(
            1,
            len(history_data["accuracy"]) + 1
        )


        # -------------------------------------------------
        # ACCURACY GRAPH
        # -------------------------------------------------

        st.write("**Training vs Validation Accuracy**")

        fig1, ax1 = plt.subplots(figsize=(8, 4))

        ax1.plot(
            epochs,
            history_data["accuracy"],
            marker="o",
            label="Training Accuracy"
        )

        ax1.plot(
            epochs,
            history_data["val_accuracy"],
            marker="o",
            label="Validation Accuracy"
        )

        ax1.set_xlabel("Epoch")

        ax1.set_ylabel("Accuracy")

        ax1.set_title("Accuracy Over Training")

        ax1.legend()

        ax1.grid(True)

        st.pyplot(fig1)

        plt.close(fig1)


        # -------------------------------------------------
        # LOSS GRAPH
        # -------------------------------------------------

        st.write("**Training vs Validation Loss**")

        fig2, ax2 = plt.subplots(figsize=(8, 4))

        ax2.plot(
            epochs,
            history_data["loss"],
            marker="o",
            label="Training Loss"
        )

        ax2.plot(
            epochs,
            history_data["val_loss"],
            marker="o",
            label="Validation Loss"
        )

        ax2.set_xlabel("Epoch")

        ax2.set_ylabel("Loss")

        ax2.set_title("Loss Over Training")

        ax2.legend()

        ax2.grid(True)

        st.pyplot(fig2)

        plt.close(fig2)


    except FileNotFoundError:

        st.info(
            "Training history is not available yet. "
            "Add training_history.json to your repository "
            "to display the graphs."
        )


# =========================================================
# NO IMAGE UPLOADED
# =========================================================

else:

    st.info(
        "👆 Upload an image above to get an AI weather prediction."
    )
