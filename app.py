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
# CUSTOM CSS
# =========================================================

st.markdown(
    """
    <style>
        .main-title {
            text-align: center;
            font-size: 42px;
            font-weight: 700;
            margin-bottom: 5px;
        }

        .subtitle {
            text-align: center;
            color: #666666;
            font-size: 18px;
            margin-bottom: 30px;
        }

        .result-card {
            padding: 25px;
            border-radius: 18px;
            border: 1px solid #dddddd;
            text-align: center;
            margin-top: 20px;
            margin-bottom: 20px;
        }

        .result-label {
            font-size: 18px;
            color: #666666;
        }

        .result-weather {
            font-size: 42px;
            font-weight: 700;
            margin: 8px 0;
        }

        .result-confidence {
            font-size: 20px;
        }

        .section-title {
            font-size: 24px;
            font-weight: 600;
            margin-top: 25px;
            margin-bottom: 15px;
        }
    </style>
    """,
    unsafe_allow_html=True
)


# =========================================================
# WEATHER INFORMATION
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

st.markdown(
    '<div class="main-title">🌦️ Weather Forecasting AI</div>',
    unsafe_allow_html=True
)

st.markdown(
    """
    <div class="subtitle">
        Upload an image of the sky and let our ResNet50 model
        classify the weather.
    </div>
    """,
    unsafe_allow_html=True
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
# IMAGE UPLOAD
# =========================================================

uploaded_file = st.file_uploader(
    "📷 Upload a sky image",
    type=["jpg", "jpeg", "png"]
)


# =========================================================
# PREDICTION
# =========================================================

if uploaded_file is not None:

    # -----------------------------------------------------
    # DISPLAY IMAGE
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
    # preprocess_input() is NOT called here.
    # Your saved model already contains:
    #
    # data_augmentation
    #        ↓
    # ResNet50 preprocess_input
    #        ↓
    # ResNet50
    #
    # Therefore raw pixel values are passed to the model.

    # -----------------------------------------------------
    # PREDICT
    # -----------------------------------------------------

    with st.spinner("🤖 Analyzing the image..."):

        predictions = model.predict(
            img_array,
            verbose=0
        )[0]

    # -----------------------------------------------------
    # GET RESULT
    # -----------------------------------------------------

    predicted_index = int(np.argmax(predictions))

    predicted_class = classes[predicted_index]

    confidence = float(predictions[predicted_index]) * 100

    icon = weather_icons[predicted_class]


    # =====================================================
    # RESULT CARD
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

    st.markdown(
        '<div class="section-title">📊 Prediction Probabilities</div>',
        unsafe_allow_html=True
    )

    # Sort classes from highest probability to lowest
    sorted_indices = np.argsort(predictions)[::-1]

    for index in sorted_indices:

        weather_class = classes[index]

        probability = float(predictions[index])

        percentage = probability * 100

        icon = weather_icons[weather_class]

        st.write(
            f"{icon} **{weather_class}** — {percentage:.2f}%"
        )

        st.progress(probability)


    # =====================================================
    # AI ANALYSIS
    # =====================================================

    st.markdown(
        '<div class="section-title">🧠 AI Analysis</div>',
        unsafe_allow_html=True
    )

    second_index = sorted_indices[1]

    second_class = classes[second_index]

    second_probability = float(
        predictions[second_index]
    ) * 100

    difference = confidence - second_probability

    st.write(
        f"The model predicts **{predicted_class}** as the most "
        f"likely weather condition with **{confidence:.2f}% "
        f"confidence**."
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

    try:

        with open("training_history.json", "r") as file:
            history_data = json.load(file)

        st.markdown(
            '<div class="section-title">📈 Training Performance</div>',
            unsafe_allow_html=True
        )

        epochs = range(
            1,
            len(history_data["accuracy"]) + 1
        )

        # -------------------------------------------------
        # ACCURACY GRAPH
        # -------------------------------------------------

        st.subheader("Training vs Validation Accuracy")

        fig1 = plt.figure(figsize=(8, 4))

        plt.plot(
            epochs,
            history_data["accuracy"],
            label="Training Accuracy"
        )

        plt.plot(
            epochs,
            history_data["val_accuracy"],
            label="Validation Accuracy"
        )

        plt.xlabel("Epoch")
        plt.ylabel("Accuracy")
        plt.title("Accuracy Over Training")
        plt.legend()
        plt.grid(True)

        st.pyplot(fig1)

        plt.close(fig1)


        # -------------------------------------------------
        # LOSS GRAPH
        # -------------------------------------------------

        st.subheader("Training vs Validation Loss")

        fig2 = plt.figure(figsize=(8, 4))

        plt.plot(
            epochs,
            history_data["loss"],
            label="Training Loss"
        )

        plt.plot(
            epochs,
            history_data["val_loss"],
            label="Validation Loss"
        )

        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title("Loss Over Training")
        plt.legend()
        plt.grid(True)

        st.pyplot(fig2)

        plt.close(fig2)

    except FileNotFoundError:

        st.info(
            "Training history is not available. "
            "Add training_history.json to the project folder "
            "to display the training graphs."
        )


# =========================================================
# NO IMAGE UPLOADED
# =========================================================

else:

    st.info(
        "👆 Upload an image above to get an AI weather prediction."
    )
