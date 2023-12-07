import tensorflow as tf
from tensorflow.keras.layers import Dense, Flatten, Conv2D
from tensorflow.keras import Model
import os
import config_loader

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
  __model:CNNModel
  __model_name:str

  def __init__(self, model:CNNModel, model_info:{}):
    self.__model=model
    self.__model_name=model_info.get('name')
    
  @property
  def model(self):
    return self.__model

  def train(self):
    mnist = tf.keras.datasets.mnist

    (x_train, y_train), (x_test, y_test) = mnist.load_data()
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

    EPOCHS = model_config.get('epochs')

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
model_config=config_loader.Loader().get_config().get('CNN-NumReader', {})
Trainer(CNNModel(model_config.get('dense_size')), model_config).train()