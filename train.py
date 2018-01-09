from keras import optimizers
from keras.layers import Dense, Activation
from keras.models import load_model
from keras.models import Sequential
from keras.utils import to_categorical
from models import simple_mlp
from util import get_mnist_data
import numpy as np
import os

def mnist_train_test_split(data, labels):
  data = data.reshape(-1, 784)
  labels = to_categorical(labels, 10)
  train_x, train_y = data[:60000], labels[:60000]
  test_x, test_y = data[60000:], labels[60000:]
  return train_x, train_y, test_x, test_y

def train_mnist_mlp():
  layer_n_units = [784, 200, 10]
  activations = ['tanh', 'softmax']
  loss = 'categorical_crossentropy'
  optimizer = 'adam'
  metrics = ['accuracy']

  data, labels = get_mnist_data()
  train_x, train_y, test_x, test_y = mnist_train_test_split(data, labels)
  train_x, test_x = train_x / 255., test_x / 255.

  model = simple_mlp(layer_n_units, activations)
  model.compile(loss=loss, optimizer=optimizer, metrics=metrics)
  model.fit(train_x, train_y, epochs=20, batch_size=128)

  print('=================')
  print(model.evaluate(test_x, test_y, batch_size=128))
  model.save('mnist_mlp.h5')

def train_mnist_mlp_autoencoder():
  layer_n_units = [784, 200, 100, 200, 784]
  activations = ['tanh', 'tanh', 'tanh', 'sigmoid']
  loss = 'mean_squared_error'
  optimizer = 'adam'
  metrics = []

  data, labels = get_mnist_data()
  train_x, train_y, test_x, test_y = mnist_train_test_split(data, labels)
  train_x, test_x = train_x / 255., test_x / 255.

  model = simple_mlp(layer_n_units, activations)
  model.compile(loss=loss, optimizer=optimizer, metrics=metrics)
  model.fit(train_x, train_x, epochs=50, batch_size=32)

  print('=================')
  print(model.evaluate(test_x, test_x, batch_size=128))
  model.save('mnist_mlp_autoencoder.h5')

def train_mnist_simple_convnet():
  from models import simple_mnist_convnet
  loss = 'categorical_crossentropy'
  optimizer = 'adam'
  metrics = ['accuracy']

  data, labels = get_mnist_data()
  train_x, train_y, test_x, test_y = mnist_train_test_split(data, labels)
  train_x, test_x = train_x / 255., test_x / 255.
  train_x = train_x.reshape(-1, 28, 28, 1)
  test_x = test_x.reshape(-1, 28, 28, 1)

  model = simple_mnist_convnet()
  model.compile(loss=loss, optimizer=optimizer, metrics=metrics)
  model.fit(train_x, train_y, epochs=10, batch_size=128)

  print('=================')
  print(model.evaluate(test_x, test_y, batch_size=128))
  model.save('mnist_simple_convnet.h5')

def eval_mnist_mlp_autoencoder():
  layer_n_units = [784, 200, 100]
  activations = ['tanh', 'tanh']
  src_model = load_model('mnist_mlp_autoencoder.h5')

  data, labels = get_mnist_data()
  train_x, train_y, test_x, test_y = mnist_train_test_split(data, labels)
  train_x, test_x = train_x / 255., test_x / 255.

  model = Sequential()
  model.add(Dense(units=layer_n_units[1], input_dim=layer_n_units[0],
                  weights=src_model.layers[0].get_weights()))
  model.add(Activation(activations[0]))
  model.add(Dense(units=layer_n_units[2],
                  weights=src_model.layers[2].get_weights()))
  model.add(Activation(activations[1]))

  activations = model.predict(test_x)
  output_dir = 'mnist_mlp_autoencoder'
  if not os.path.exists(output_dir):
    os.mkdir(output_dir)
  np.save(output_dir + '/mnist_mlp_autoencoder_activations.npy', activations)
  np.save(output_dir + '/mnist_test_data.npy',
          np.stack([data[60000:]] * 3, axis=3))
  np.save(output_dir + '/mnist_test_data_flattened.npy', test_x)
  np.save(output_dir + '/mnist_test_labels.npy', labels[60000:].reshape(-1, 1))

if __name__ == '__main__':
  train_mnist_mlp()
  train_mnist_simple_convnet()
  train_mnist_mlp_autoencoder()
  eval_mnist_mlp_autoencoder()
