import streamlit as st
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model

# Load the pre-trained deep learning model for hand digit recognition
model = load_model('digit_recognition_model.h5')

# Title of the web application
st.title("Hand Digit Recognition")

# Instructions
st.write("""
    This app recognizes hand-written digits.
    Draw a digit in the canvas below and click on "Classify" to get the prediction.
""")

# Create a canvas for user input
canvas = st.canvas(
    fill_color="white",
    stroke_color="black",
    background_color="white",
    width=280,
    height=280,
    drawing_mode="freedraw",
    key="canvas",
)

# Button to make prediction
if st.button("Classify"):
    if canvas.image_data is not None:
        # Preprocess the image
        img_array = np.array(canvas.image_data)
        
        # Convert to grayscale and normalize the image
        img_gray = np.mean(img_array, axis=-1)  # Convert to grayscale
        img_resized = np.resize(img_gray, (28, 28))  # Resize to 28x28 pixels
        
        # Normalize the image
        img_resized = img_resized / 255.0
        
        # Reshape to match the input shape of the model (28, 28, 1)
        img_input = np.expand_dims(img_resized, axis=-1)
        img_input = np.expand_dims(img_input, axis=0)
        
        # Make prediction using the trained model
        prediction = model.predict(img_input)
        
        # Show the result
        predicted_digit = np.argmax(prediction)
        st.success(f"The predicted digit is: {predicted_digit}")

        # Show the drawn image
        st.image(img_resized, caption='Drawn Image', use_column_width=True)

    else:
        st.error("Please draw a digit on the canvas.")
