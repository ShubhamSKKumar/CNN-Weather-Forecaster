import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

st.title("Weather Forecasting AI 🌦️")
st.write("Upload an image of the sky, and the ResNet50 AI will predict the weather!")

@st.cache_resource
def load_model():
    return tf.keras.models.load_model(
        'models/02_resnet50_weather_classifier.keras',
        compile=False
    )
    
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
