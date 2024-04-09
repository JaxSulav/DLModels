import cv2
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array

model = load_model('models/face_cls_3.2.h5')
label_maps = {0: 'Ahegao', 1: 'Angry', 2: 'Happy', 3: 'Neutral', 4: 'Sad', 5: 'Surprise'}


def preprocess_image(image_path):
    # Read the image
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) 
    img = cv2.resize(img, (96, 96)) 
    img = img_to_array(img)
    img = np.expand_dims(img, axis=0)
    img /= 255.0  
    return img

def predict_expression(image_path):
    preprocessed_img = preprocess_image(image_path)
    
    prediction = model.predict(preprocessed_img)
    return np.argmax(prediction, axis=1), prediction
    

image_path = './jennifer.jpg'
expression_id, probabilities = predict_expression(image_path)
exp = label_maps[expression_id[0]]
print(f'Predicted Expression: {exp}')