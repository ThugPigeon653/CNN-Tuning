import tensorflow as tf
from tensorflow.keras.layers import Dense, Flatten, Conv2D
from tensorflow.keras import Model
import os
import requests

os.chdir(os.path.dirname(os.path.abspath(__file__)))

class CNNModel(tf.keras.Model):
  def __init__(self, dense_size:int):
    super().__init__()
    self.conv1 = Conv2D(32, 3, activation='relu')
    self.flatten = Flatten()
    self.d1 = Dense(dense_size, activation='relu')
    self.d2 = Dense(10)

  def call(self, x):
    x = self.conv1(x)
    x = self.flatten(x)
    x = self.d1(x)
    return self.d2(x)
  
class Trainer():
  __train_data_endpoint:str
  __model:CNNModel
  __model_name:str

  def __init__(self, model:CNNModel):
    self.__train_data_endpoint=os.environ.get('TRAIN_DATA_ENDPOINT')+':'+os.environ.get('TRAIN_DATA_PORT')
    print(f"ENDPOINT: {self.__train_data_endpoint}")
    self.__model=model
    self.__model_name='CNN-NumReader'
    
  @property
  def model(self):
    return self.__model

  def train(self, epochs:int=3):
    try:
      response = requests.get(self.__train_data_endpoint)
    except Exception as e:
      print(f"Failed to get mnist data at: {self.__train_data_endpoint}\n{e}")
    if response.status_code == 200:
        data = response.json()
        print(f"\n\n\n{data}\n\n\n")
        x_train, y_train = data['x_train'], data['y_train']
        x_test, y_test = data['x_test'], data['y_test']
    else:
      raise Exception(f"Could not load training data\n{response.json()}")
    x_train, x_test = x_train / 255.0, x_test / 255.0

    x_train = x_train[..., tf.newaxis].astype("float32")
    x_test = x_test[..., tf.newaxis].astype("float32")

    train_ds = tf.data.Dataset.from_tensor_slices(
        (x_train, y_train)).shuffle(10000).batch(32)

    test_ds = tf.data.Dataset.from_tensor_slices((x_test, y_test)).batch(32)

    loss_object = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)

    optimizer = tf.keras.optimizers.Adam()
    train_loss = tf.keras.metrics.Mean(name='train_loss')
    train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='train_accuracy')

    test_loss = tf.keras.metrics.Mean(name='test_loss')
    test_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='test_accuracy')
    @tf.function
    def train_step(images, labels):
      with tf.GradientTape() as tape:
        predictions = self.__model(images, training=True)
        loss = loss_object(labels, predictions)
      gradients = tape.gradient(loss, self.__model.trainable_variables)
      optimizer.apply_gradients(zip(gradients, self.__model.trainable_variables))

      train_loss(loss)
      train_accuracy(labels, predictions)
    @tf.function
    def test_step(images, labels):
      predictions = self.__model(images, training=False)
      t_loss = loss_object(labels, predictions)

      test_loss(t_loss)
      test_accuracy(labels, predictions)

    EPOCHS = epochs

    for epoch in range(EPOCHS):
      # Reset the metrics at the start of the next epoch
      train_loss.reset_states()
      train_accuracy.reset_states()
      test_loss.reset_states()
      test_accuracy.reset_states()

      for images, labels in train_ds:
        train_step(images, labels)

      for test_images, test_labels in test_ds:
        test_step(test_images, test_labels)

      print(
        f'Epoch {epoch + 1}, '
        f'Loss: {train_loss.result()}, '
        f'Accuracy: {train_accuracy.result() * 100}, '
        f'Test Loss: {test_loss.result()}, '
        f'Test Accuracy: {test_accuracy.result() * 100}'
      )

    self.__model.save(self.__model_name)

# implementation
try:
    response = requests.get(f"{os.environ.get('CONFIG_DATA_ENDPOINT')}:{os.environ.get('CONFIG_DATA_PORT')}")
    #response.raise_for_status() 
    data = response.json()['CNN-NumReader']
    print("Request successful. Data received:", data)
    

except requests.exceptions.RequestException as e:
      print('\n\n\n\n')
      print('==================')
      print(f"content: {response.content}\ntext: {response.text}")
      print('\n\n\n\n')

Trainer(CNNModel(data['dense_size'])).train()