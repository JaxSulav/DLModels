import streamlit as st
from keras.models import load_model
from keras.preprocessing import image
import numpy as np
import gdown
import os

def load_prediction_model():
    os.makedirs("models", exist_ok=True)
    model_path = 'models/cnn_exp_cls.h5'
    if not os.path.isfile(model_path):
        url = 'https://drive.google.com/uc?export=download&id=1hPckCu2iry__4cIoHiqKjdnHx020bZqY'
        gdown.download(url, model_path, quiet=False)
    return load_model(model_path)

model = load_prediction_model()

def prepare_image(img):
    img_array = image.img_to_array(img) 
    img_array_expanded = np.expand_dims(img_array, axis=0)  
    return img_array_expanded / 255. 

st.title('Facial Expression Classifier with CNN Model')
st.write("Upload an image and the classifier will predict the facial expression.")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    img = image.load_img(uploaded_file, target_size=(64, 64))
    st.write("")
    st.write("Classifying...")
    label_maps = {0: 'Angry', 1: 'Ahegao', 2: 'Happy', 3: 'Neutral', 4: 'Sad', 5: 'Surprise'}
    result = model.predict(prepare_image(img))
    exp = label_maps[np.argmax(result)]
    st.write(f'Predicted class for the image: {exp}')
    st.image(img, caption='Uploaded Image', use_column_width=True)
