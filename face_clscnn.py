from keras.models import load_model
from keras.preprocessing import image
import numpy as np
import gdown
import os

os.makedirs("models", exist_ok=True)
if not os.path.isfile('models/cnn_exp_cls.h5'):
    url = 'https://drive.google.com/uc?export=download&id=1hPckCu2iry__4cIoHiqKjdnHx020bZqY'
    output = 'models/cnn_exp_cls.h5'
    gdown.download(url, output, quiet=False)
else:
    print("Model already exists")

model = load_model('models/cnn_exp_cls.h5')  

def prepare_image(img_path):
    img = image.load_img(img_path, target_size=(64, 64))  
    img_array = image.img_to_array(img) 
    img_array_expanded = np.expand_dims(img_array, axis=0) 
    return img_array_expanded / 255.  

def classify_image(img_path):
    prepared_img = prepare_image(img_path)
    prediction = model.predict(prepared_img)
    return np.argmax(prediction)  

label_maps = {0: 'Ahegao', 1: 'Angry', 2: 'Happy', 3: 'Neutral', 4: 'Sad', 5: 'Surprise'}
img_path = 'jennifer.jpg'  
result = classify_image(img_path)
exp = label_maps[result]
print(f'Predicted class for the image: {exp}')
