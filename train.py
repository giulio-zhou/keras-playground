from models import simple_mlp
from util import get_mnist_data
import numpy as np

def train_mnist_mlp():
  layer_n_units = [784, 100, 10]
  activations = ['tanh', 'softmax']
  loss = 'categorical_crossentropy'
  optimizer = 'adam'
  metrics = ['accuracy']

  data, labels = get_mnist_data()
  data = data.reshape(-1, 784)
  labels = np.diag(np.arange(10))[labels.astype(np.uint8)]
  print(labels)
  train_x, train_y = data[:60000], labels[:60000]
  test_x, test_y = data[60000:], labels[60000:]

  model = simple_mlp(layer_n_units, activations)
  model.compile(loss=loss, optimizer=optimizer, metrics=metrics)
  model.fit(train_x, train_y, epochs=20, batch_size=32)

  print('=================')
  print(model.evaluate(test_x, test_y, batch_size=128))
  model.save('mnist_mlp.h5')

if __name__ == '__main__':
  train_mnist_mlp()
