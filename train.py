from keras import optimizers
from keras.layers import Dense, Activation
from keras.models import load_model
from keras.models import Sequential
from keras.utils import to_categorical
from models import make_model_trainable 
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

def train_mnist_conv_autoencoder():
  from models import mnist_conv_autoencoder
  loss = 'mean_squared_error'
  optimizer = 'adam'
  metrics = []

  data, labels = get_mnist_data()
  train_x, train_y, test_x, test_y = mnist_train_test_split(data, labels)
  train_x, test_x = train_x / 255., test_x / 255.
  train_x = train_x.reshape(-1, 28, 28, 1)
  test_x = test_x.reshape(-1, 28, 28, 1)

  model = mnist_conv_autoencoder()
  model.compile(loss=loss, optimizer=optimizer, metrics=metrics)
  model.fit(train_x, train_x, epochs=50, batch_size=32)

  print('=================')
  print(model.evaluate(test_x, test_x, batch_size=128))
  model.save('mnist_conv_autoencoder.h5')

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

def train_gan(g, d, GAN, X_train, num_epochs=20, batch_size=32, input_size=50,
              num_discriminator_iters=10, pretrain_discriminator=False):
  if pretrain_discriminator:
    num_pretrain_ex = 10000
    idx = np.random.choice(np.arange(len(X_train)), num_pretrain_ex)
    img_batch = X_train[idx]
    input_noise = np.random.normal(0, 1, (num_pretrain_ex, input_size))
    generated_images = g.predict(input_noise)
    X = np.concatenate([img_batch, generated_images])
    y = np.zeros([2 * num_pretrain_ex, 1])
    y[num_pretrain_ex:] = 1 # Fakes are 1's
    d_loss = d.fit(X, y, epochs=1, batch_size=batch_size)

  # Start joint training.
  num_iter = num_epochs * len(X_train) // batch_size
  for i in range(num_iter):
    for k in range(num_discriminator_iters):
      # Make generator images.
      idx = np.random.choice(np.arange(len(X_train)), batch_size)
      img_batch = X_train[idx]
      input_noise = np.random.normal(0, 1, (batch_size, input_size))
      generated_images = g.predict(input_noise)
      # Train discriminator.
      make_model_trainable(d, True)
      # 1's are fakes.
      d_loss_real = d.train_on_batch(img_batch, np.zeros((batch_size, 1)))
      d_loss_fake = d.train_on_batch(generated_images,
                                     np.ones((batch_size, 1)))
      d_loss = 0.5 * (d_loss_real + d_loss_fake)
    # Train generator-discriminator directly from input noise.
    noise = np.random.normal(0, 1, (batch_size, input_size))
    make_model_trainable(d, False)
    # Discriminator should think that these are real.
    g_loss = GAN.train_on_batch(noise, np.zeros((batch_size, 1)))
    if i % (len(X_train) // 100) == 0:
       print(i, d_loss, g_loss)
       print(d_loss_real, d_loss_fake)
  return g, d

def train_gan_output_generated_imgs(g, train_gan_fn, num_epochs, output_every,
                                    input_size, num_imgs):
  for i in range(0, num_epochs, output_every):
    num_training_epochs = min(output_every, num_epochs - i)
    train_gan_fn(num_training_epochs)
    input_noise = np.random.normal(0, 1, (num_imgs, input_size))
    after_n_epochs_samples = g.predict(input_noise)
    np.save('after_%i_epochs_samples.npy' % (i + num_training_epochs),
            after_n_epochs_samples)

def train_simple_mnist_gan():
  from models import mnist_simple_gan
  g, d, GAN = mnist_simple_gan()

  data, labels = get_mnist_data()
  train_x, train_y, test_x, test_y = mnist_train_test_split(data, labels)
  train_x, test_x = train_x / 255., test_x / 255.
  train_x, test_x = 2 * (train_x - 0.5), 2 * (test_x - 0.5)

  from keras.optimizers import Adam, SGD
  adam_optimizer = Adam(1e-3)
  sgd_optimizer = SGD(0.001)
  g.compile(loss='binary_crossentropy', optimizer=adam_optimizer)
  d.compile(loss='binary_crossentropy', optimizer=adam_optimizer)
  GAN.compile(loss='binary_crossentropy', optimizer=adam_optimizer)

  train_gan_fn = lambda num_epochs: train_gan(g, d, GAN, train_x, num_epochs,
                                              batch_size=32, input_size=100)
  train_gan_output_generated_imgs(g, train_gan_fn, 30, 5, 100, 16)

def train_mnist_conv_gan():
  from models import mnist_conv_gan
  g, d, GAN = mnist_conv_gan()

  data, labels = get_mnist_data()
  train_x, train_y, test_x, test_y = mnist_train_test_split(data, labels)
  train_x, test_x = train_x / 255., test_x / 255.
  train_x, test_x = 2 * (train_x - 0.5), 2 * (test_x - 0.5)
  train_x = train_x.reshape(-1, 28, 28, 1)
  test_x = test_x.reshape(-1, 28, 28, 1)

  from keras.optimizers import Adam, SGD
  adam_optimizer = Adam()
  sgd_optimizer = SGD(0.001)
  g.compile(loss='binary_crossentropy', optimizer=adam_optimizer)
  d.compile(loss='binary_crossentropy', optimizer=adam_optimizer)
  GAN.compile(loss='binary_crossentropy', optimizer=adam_optimizer)

  train_gan_fn = lambda num_epochs: train_gan(g, d, GAN, train_x, num_epochs,
                                              batch_size=128, input_size=100,
                                              num_discriminator_iters=3)
  train_gan_output_generated_imgs(g, train_gan_fn, 30, 5, 100, 16)

if __name__ == '__main__':
  train_mnist_mlp()
  train_mnist_simple_convnet()
  train_mnist_mlp_autoencoder()
  eval_mnist_mlp_autoencoder()
  train_mnist_conv_autoencoder()
  train_simple_mnist_gan()
  train_mnist_conv_gan()
