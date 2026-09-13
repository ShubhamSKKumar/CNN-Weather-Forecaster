import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

st.title("Weather Forecasting AI 🌦️")
st.write("Upload an image of the sky, and the ResNet50 AI will predict the weather!")

# Cache the model so it doesn't reload on every click
@st.cache_resource
@st.cache_resource
def load_model():
    weights_path = 'models/02_resnet50_weather_classifier.h5'
    
    try:
        # Architecture 1: The Nested Sequential (matches a 4-layer topology)
        data_augmentation = tf.keras.Sequential([
            tf.keras.layers.Rescaling(1./255),
            tf.keras.layers.RandomFlip("horizontal"),
            tf.keras.layers.RandomRotation(0.2),
            tf.keras.layers.RandomZoom(0.2),
        ], name="data_augmentation")

        model = tf.keras.Sequential([
            data_augmentation,
            tf.keras.applications.ResNet50(weights=None, include_top=False, input_shape=(224, 224, 3)),
            tf.keras.layers.GlobalAveragePooling2D(),
            tf.keras.layers.Dense(4, activation='softmax')
        ])
        model.build((None, 224, 224, 3))
        model.load_weights(weights_path)
        return model
        
    except ValueError:
        # Architecture 2: The Flat Sequential (matches a 7-layer topology)
        model = tf.keras.Sequential([
            tf.keras.layers.Rescaling(1./255, input_shape=(224, 224, 3)),
            tf.keras.layers.RandomFlip("horizontal"),
            tf.keras.layers.RandomRotation(0.2),
            tf.keras.layers.RandomZoom(0.2),
            tf.keras.applications.ResNet50(weights=None, include_top=False),
            tf.keras.layers.GlobalAveragePooling2D(),
            tf.keras.layers.Dense(4, activation='softmax')
        ])
        model.build((None, 224, 224, 3))
        model.load_weights(weights_path)
        return model

model = load_model()
classes = ['Cloudy', 'Rain', 'Shine', 'Sunrise'] 

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert('RGB')
    st.image(image, caption='Uploaded Image', use_column_width=True)

    st.write("Classifying...")
    img = image.resize((224, 224))
    img_array = tf.keras.preprocessing.image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0) 

    predictions = model.predict(img_array)[0]
    predicted_class = classes[np.argmax(predictions)]
    confidence = np.max(predictions) * 100

    st.success(f"Prediction: **{predicted_class}** ({confidence:.2f}%)")
