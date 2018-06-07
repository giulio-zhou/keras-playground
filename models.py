from keras.layers import Conv2D, Conv2DTranspose
from keras.layers import Dense, Activation
from keras.layers import Input
from keras.layers import Flatten, Reshape
from keras.layers import BatchNormalization
from keras.layers.advanced_activations import LeakyReLU
from keras.models import Model, Sequential

def simple_mlp(layer_n_units, activations, weights=[]):
  """
  layer_n_units: list containing number of units in each layer (length: N)
  activations: list of names of activations for each layer (length: N-1)
  weights: (optional) list of weights to initialize dense layers with.
  """
  assert len(layer_n_units) - 1 == len(activations)
  assert len(layer_n_units) >= 1
  assert len(weights) == 0 or len(weights) == len(layer_n_units) - 1
  weight_matrix = weights[0] if len(weights) > 0 else None
  model = Sequential() 
  model.add(Dense(units=layer_n_units[1], input_dim=layer_n_units[0],
                  weights=weight_matrix))
  model.add(Activation(activations[0]))
  for i in range(2, len(layer_n_units)):
    weight_matrix = weights[i-1] if len(weights) > 0 else None
    model.add(Dense(units=layer_n_units[i], weights=weight_matrix))
    model.add(Activation(activations[i-1]))
  return model

def simple_mnist_convnet():
  model = Sequential()
  model.add(Conv2D(8, 3, strides=(1, 1), padding="same",
                   input_shape=(28, 28, 1)))
  model.add(Activation('relu'))
  model.add(Conv2D(8, 3, strides=(2, 2), padding="same"))
  model.add(Activation('relu'))
  model.add(Conv2D(16, 3, strides=(1, 1), padding="same"))
  model.add(Activation('relu'))
  model.add(Conv2D(16, 3, strides=(2, 2), padding="same"))
  model.add(Activation('relu'))
  model.add(Flatten())
  model.add(Dense(units=200))
  model.add(Activation('relu'))
  model.add(Dense(units=10))
  model.add(Activation('softmax'))
  return model

def mnist_conv_autoencoder():
  model = Sequential()
  model.add(Conv2D(16, 3, strides=(2, 2), padding="same",
                   input_shape=(28, 28, 1)))
  model.add(Activation('relu'))
  model.add(Conv2D(32, 3, strides=(2, 2), padding="same"))
  model.add(Activation('relu'))
  model.add(Flatten())
  model.add(Dense(units=500))
  model.add(Activation('relu'))
  model.add(Dense(units=200))
  model.add(Activation('relu'))
  model.add(Dense(units=100))
  model.add(Activation('relu'))
  model.add(Dense(units=200))
  model.add(Activation('relu'))
  model.add(Dense(units=7*7*32))
  model.add(Activation('relu'))
  model.add(Reshape((7, 7, 32)))
  model.add(Conv2DTranspose(16, 3, strides=(2, 2), padding="same"))
  model.add(Activation('relu'))
  model.add(Conv2DTranspose(1, 3, strides=(2, 2), padding="same"))
  model.add(Activation('sigmoid'))
  return model

# Utility function for training GANs.
def make_model_trainable(model, val):
  model.trainable = val
  for l in model.layers:
      l.trainable = val

def make_stacked_gan(g, d, gan_input):
  generator_output = g(gan_input)
  discriminator_output = d(generator_output)
  GAN = Model(gan_input, discriminator_output)
  return GAN

def mnist_simple_gan():
  g = simple_mlp([100, 200, 784], ['tanh', 'tanh'])
  d = simple_mlp([784, 200, 1], ['tanh', 'sigmoid'])
  GAN = make_stacked_gan(g, d, Input(shape=(100,)))
  return g, d, GAN

def mnist_conv_gan():
  # Generator model.
  g = Sequential()
  g.add(Dense(units=400, input_dim=100))
  g.add(Activation('relu'))
  g.add(Reshape((5, 5, 16)))
  g.add(Conv2DTranspose(16, 3, strides=(1, 1), padding="valid"))
  g.add(Activation('relu'))
  g.add(Conv2DTranspose(8, 5, strides=(2, 2), padding="same"))
  g.add(Activation('relu'))
  g.add(Conv2DTranspose(1, 5, strides=(2, 2), padding="same"))
  g.add(Activation('tanh'))
  # Discriminator model.
  d = Sequential()
  d.add(Conv2D(8, 3, strides=(1, 1), padding="same",
               input_shape=(28, 28, 1)))
  d.add(Activation('relu'))
  d.add(Conv2D(8, 3, strides=(2, 2), padding="same"))
  d.add(Activation('relu'))
  d.add(Conv2D(16, 3, strides=(1, 1), padding="same"))
  d.add(Activation('relu'))
  d.add(Conv2D(16, 3, strides=(2, 2), padding="same"))
  d.add(Activation('relu'))
  d.add(Flatten())
  d.add(Dense(units=200))
  d.add(Activation('relu'))
  d.add(Dense(units=1))
  d.add(Activation('sigmoid'))
  # d = Sequential()
  # d.add(Flatten(input_shape=(28, 28, 1)))
  # d.add(Dense(units=200, input_shape=(784,)))
  # d.add(Activation('tanh'))
  # d.add(Dense(units=1))
  # d.add(Activation('sigmoid'))
  # Stacked GAN model.
  GAN = make_stacked_gan(g, d, Input(shape=(100,)))
  return g, d, GAN
