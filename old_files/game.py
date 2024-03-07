import random
import numpy as np
from motor import Value
from hjernen import Neuron, Layer, MLP
import pandas as pd

# initialize a model 
model = MLP(7, [16, 16, 16, 1]) # x-layer neural network
model.load_weights('weights.json')

def predict(input_data):
    # Convert input_data to Value objects
    input_values = [list(map(Value, input_data))]

    # Forward pass to get predicted scores
    scores = list(map(model, input_values))
    #print(scores)
    # Return the predicted scores
    return [score.data for score in scores]

new_input = [18, 5, 4, 1, 2, 3, 1]  # Example input data
prediction = predict(new_input)
print("Predicted Score:", prediction)