from flask import Flask, jsonify
import tensorflow as tf
import numpy as np
import boto3

(x_train, y_train),(x_test, y_test) = tf.keras.datasets.mnist.load_data()

app = Flask(__name__)

def load_images_from_s3(bucket, s3_client):
    response = s3_client.list_objects(Bucket=bucket)
    images = []
    for obj in response.get('Contents', []):
        image_content = s3_client.get_object(Bucket=bucket, Key=obj['Key'])['Body'].read()
        image = np.frombuffer(image_content, dtype=np.uint8).reshape((28, 28))
        images.append(image.tolist())
    return images

@app.route('/', methods=["GET"])
def get_training_data():
    try:    
        s3_client = boto3.client('s3')
        response = s3_client.list_objects('tensorflow_tuning_cnn_data')
        num_objects = len(response.get('Contents', []))
        if(num_objects>0):
            (x_train, y_train),(x_test, y_test)=load_images_from_s3('tensorflow_tuning_cnn_data')   
        else:
            (x_train, y_train),(x_test, y_test) = tf.keras.datasets.mnist.load_data()

        mnist_data = {'x_train': x_train.tolist(), 'y_train': y_train.tolist(), 'x_test':x_test.tolist(), 'y_test':y_test.tolist()}
        print("DATASET LOADED")
        return jsonify(mnist_data)
    except Exception as e:
        print(f"Error loading training data: {e}")
        return jsonify({'error': 'Failed to load training data'}), 500

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000)
