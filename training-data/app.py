from flask import Flask, jsonify
import tensorflow as tf
import numpy as np

(x_train, y_train),(x_test, y_test) = tf.keras.datasets.mnist.load_data()

app = Flask(__name__)

@app.route('/', methods=["GET"])
def get_training_data():
    try:
        (x_train, y_train),(x_test, y_test) = tf.keras.datasets.mnist.load_data()

        mnist_data = {'x_train': x_train.tolist(), 'y_train': y_train.tolist(), 'x_test':x_test.tolist(), 'y_test':y_test.tolist()}
        print("DATASET LOADED")
        return jsonify(mnist_data)
    except Exception as e:
        print(f"Error loading training data: {e}")
        return jsonify({'error': 'Failed to load training data'}), 500

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000)
