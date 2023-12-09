from flask import Flask, jsonify
import tensorflow as tf

app = Flask(__name__)

@app.route('/', methods=["GET"])
def get_training_data():
    try:
        (x_train, y_train), _ = tf.keras.datasets.mnist.load_data()

        mnist_data = {'x_train': x_train.tolist(), 'y_train': y_train.tolist()}
        print("DATASET LOADED")
        return jsonify(mnist_data)
    except Exception as e:
        print(f"Error loading training data: {e}")
        return jsonify({'error': 'Failed to load training data'}), 500

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000)
