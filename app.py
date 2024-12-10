
import streamlit as st
import numpy as np
import tensorflow as tf
import cv2
model = tf.keras.models.load_model('path_to_save_model\my_model.h5')
st.header('Hand Digit Recognition Model')
img= st.text_input('Enter Image Name')
image = cv2.imread(img)[:,:,0]
image = np.invert(np.array([image]))
output=model.predict(image)
stn = 'Digit in the Image is '+ str(np.argmax(output))
st.markdown(stn)