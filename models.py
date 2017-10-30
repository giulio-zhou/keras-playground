from keras.layers import Dense, Activation
from keras.models import Sequential

def simple_mlp(layer_n_units, activations):
  """
  layer_n_units: list containing number of units in each layer (length: N)
  activations: list of names of activations for each layer (length: N-1)
  """
  assert len(layer_n_units) - 1 == len(activations)
  assert len(layer_n_units) >= 1
  model = Sequential() 
  model.add(Dense(units=layer_n_units[1], input_dim=layer_n_units[0]))
  model.add(Activation(activations[0]))
  for i in range(2, len(layer_n_units)):
    model.add(Dense(units=layer_n_units[i]))
    model.add(Activation(activations[i-1]))
  return model
