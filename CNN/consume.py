import tensorflow as tf
from tensorflow.keras.models import load_model
from PIL import Image
import numpy as np
import config_loader

class Predictions():
    def __init__():
        model = load_model(config_loader.Loader().get_config().get('model', {}).get('name'))
        img_path = "image.jpg"
        img = Image.open(img_path).convert("L") 
        img = img.resize((28, 28))
        img_array = np.array(img)
        img_array = img_array[np.newaxis, ..., np.newaxis].astype("float32")  
        img_array = img_array / 255.0  

        model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
        predictions = model.predict(img_array)
        predicted_class = np.argmax(predictions[0])

        print("Predicted Class:", predicted_class)

