from flask import Flask
import tensorflow as tf

@app.route('/', methods=["GET"])
def get_training_data():
    return tf.keras.datasets.mnist