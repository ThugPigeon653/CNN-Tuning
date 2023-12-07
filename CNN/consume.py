import tensorflow as tf
from tensorflow.keras.models import load_model
from PIL import Image
import numpy as np
import config_loader
import os

os.chdir(os.path.dirname(os.path.abspath(__file__)))

class Predictions():
    @staticmethod
    def predict(model_name:str)->int:
        model = load_model(config_loader.Loader().get_config().get(model_name, {}).get('name'))
        img_path = "image.jpg"
        img = Image.open(img_path).convert("L") 
        img = img.resize((28, 28))
        img_array = np.array(img)
        img_array = img_array[np.newaxis, ..., np.newaxis].astype("float32")  
        img_array = img_array / 255.0  

        predictions = model.predict(img_array)
        predicted_class = np.argmax(predictions[0])

        return predicted_class

print(Predictions().predict('CNN-NumReader'))