import random
import numpy as np
from motor import Value
from hjernen import Neuron, Layer, MLP
import pandas as pd
import math


from ucimlrepo import fetch_ucirepo 
# https://archive.ics.uci.edu/dataset/320/student+performance
  
# fetch dataset 
student_performance = fetch_ucirepo(id=320) 
selected_features = student_performance.data.features[['age', 'Medu', 'Fedu', 'traveltime', 'studytime', 'goout', 'Dalc']] # famsup og sex er fjernet for nu
# data (as pandas dataframes) 
X = pd.DataFrame(selected_features).values # Inputs
y = pd.DataFrame(student_performance.data.targets).iloc[:, -1].values # Only take the last coloumn of grades. The G3. y is goals. 

# initialize a model 
model = MLP(7, [10, 10, 10, 20]) # The neural network
#model.load_weights('weights.json') 
print("number of parameters", len(model.parameters()))

# loss function
def loss(batch_size=None):
    # inline DataLoader :)
    if batch_size is None: # use the entire dataset (X, y) without batching.
        Xb, yb = X, y
    else: # Choose a batch of random values
        ri = np.random.permutation(X.shape[0])[:batch_size]
        Xb, yb = X[ri], y[ri]
    inputs = [list(map(Value, xrow)) for xrow in Xb]

    # forward the model 
    prediction = list(map(model, inputs)) # get prediction for each output neuron.

    # softmax loss
    exp_prediction = [[np.exp(predi.data) for predi in sublist] for sublist in prediction] # Take exponential of each prediction datapoint 
    sum_exp_scores = [np.sum(sublist) for sublist in exp_prediction] # The sum of prediction numbers. 
    softmax_probs = np.array([[element / sublist_sum for element in sublist] for sublist, sublist_sum in zip(exp_prediction, sum_exp_scores)]) # Normalize to percentage. The sum is 100%. prediction gets a percentage. datapoint in sublist / sum of datapoints in sublist.  A numpy list of probabilities for each grade based on the inputs.

    # one-hot encode the target labels, 1 at the correct answer. (the grade)
    y_onehot = np.zeros_like(softmax_probs) # Create numpy array with same dimensions as softmax_probs
    y_onehot[np.arange(len(yb)), yb] = 1 # Insert 1 in the index where the correct answer is
    #print(softmax_probs[:10, :])  
    #print(len(y_onehot))
    #print(y_onehot[:10, :])  

    # cross-entropy loss
    # Clip a probability to avoid to low or too high numbers as its bad for log function
    softmax_probs_clipped = np.clip(softmax_probs, 1e-7, 1 - 1e-7)
    cross_entropy_loss = -np.sum(y_onehot * np.log(softmax_probs_clipped), axis=1) ## Find the certency of the right answer and take log of it. Finds for each row. Right answer * models probability that this is the right answer. Sum because of one hot. 1*0.3+0*0.3+0*0.3 = 0.3. 
    average_loss = np.mean(cross_entropy_loss) 

    # L2 regularization
    alpha = 1e-4
    reg_loss = alpha * sum((p*p for p in model.parameters()))
    total_loss = average_loss + reg_loss # How well the neural net is performing

    # also get accuracy
    predicted_labels = np.argmax(softmax_probs, axis=1, keepdims=True)
    accuracy = np.mean(predicted_labels == yb) # Check if predicted is equal to actual value. 
    return total_loss, accuracy, 

#total_loss, acc = loss()
#print(total_loss, acc)

# optimization
for k in range(5):
    
    # forward
    total_loss, acc = loss()
    
    # backward
    model.zero_grad()
    total_loss.backward()
    learning_rate = 1.0 - 0.9*k/100
    for p in model.parameters():
        p.data -= learning_rate * p.grad
    if k % 1 == 0:
        print(f"step {k} loss {total_loss.data}, accuracy {acc*100}%")
model.save_weights('weights.json')


# Sex
# Age
# Famsize
# Pstatus - Er dine forældre skilt
# Medu - Mother education, 0-4
# Fedu - Father education, 0-4
# guardian - Måske. Her skal man sige om ens guardian er far eller mor.
# traveltime - Hvor langt har du til skole
# studytime - Hvor mange timer bruger du på uni
# famsup Støtter din familie dig
# romantic - Er du i en romantisk relation
# goout - Hvor ofte tager du i byen
# Dalc - Hvor mange øl drikker du om dage?


