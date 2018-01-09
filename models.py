from keras.layers import Dense, Activation
from keras.models import Sequential

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
