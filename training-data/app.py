from flask import Flask
import tensorflow as tf

app=Flask(__name__)

@app.route('/', methods=["GET"])
def get_training_data():
    return tf.keras.datasets.mnist