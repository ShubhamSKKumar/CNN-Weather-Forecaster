import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

st.title("Weather Forecasting AI 🌦️")
st.write("Upload an image of the sky, and the ResNet50 AI will predict the weather!")

# Cache the model so it doesn't reload on every click
@st.cache_resource
def load_model():
    # 1. Rebuild the exact empty ResNet50 architecture you trained
    base_model = tf.keras.applications.ResNet50(weights=None, include_top=False, input_shape=(224, 224, 3))
    x = tf.keras.layers.GlobalAveragePooling2D()(base_model.output)
    output = tf.keras.layers.Dense(4, activation='softmax')(x)
    model = tf.keras.models.Model(inputs=base_model.input, outputs=output)
    
    # 2. Inject only the learned mathematical weights from your saved file
    model.load_weights('models/02_resnet50_weather_classifier.h5')
    
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
