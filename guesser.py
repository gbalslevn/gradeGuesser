from motor import Value
from hjernen import Neuron, Layer, MLP
import numpy as np

# initialize a model 
model = MLP(7, [2, 20]) # x-layer neural network
model.load_weights('weights.json')

def predict(input_data):
    # Convert input_data to Value objects
    input_values = [list(map(Value, input_data))]

    # Forward pass to get predicted scores
    predictions = list(map(model, input_values))
    # Return the predicted scores
    return predictions

new_input = [18, 5, 4, 1, 2, 3, 1]  # Example input data
prediction = predict(new_input)
# softmax loss
exp_prediction = [[np.exp(predi.data) for predi in sublist] for sublist in prediction] # Take exponential of each prediction datapoint 
sum_exp_scores = [np.sum(sublist) for sublist in exp_prediction] # The sum of prediction numbers. 
softmax_probs = np.array([[element / sublist_sum for element in sublist] for sublist, sublist_sum in zip(exp_prediction, sum_exp_scores)]) # Normalize to percentage. The sum is 100%. prediction gets a percentage. datapoint in sublist / sum of datapoints in sublist.  A numpy list of probabilities for each grade based on the inputs.

predicted_labels = np.argmax(softmax_probs, axis=1, keepdims=True)
score = predicted_labels[0][0]

print("Predicted Score:", score)