import random
import numpy as np
from motor import Value
from hjernen import Neuron, Layer, MLP

model = MLP(3, [4, 4, 1]) # 2-layer neural network
# Four inputs
xs = [
    [2.0, 3.0, -1.0],
    [3.0, -1.0, 0.5],
    [0.5, 1.0, 1.0],
    [1.0, 1.0, -1.0],
]
# Based on first row input first row output should be 1. Second row -1 and so on.
ys = [1.0, -1.0, -1.0, 1.0] # goal
print("number of parameters", len(model.parameters()))

# Gradient descent
for k in range(10):
    # forward pass
    ypred = [model(x) for x in xs] # result after inputs and weights
    # We need to tune weights to achieve the goal ys
    loss = sum([(yout - ygt)**2 for ygt, yout in zip(ys, ypred)]) # measures how well the neural net is performing. Y ground truth. result - goal for each neuron. summed we get the loss 
    
    # backward pass
    for p in model.parameters():
        p.grad = 0.0 # Start at zero again. Because we start at the top of the tree and run down so every node will get a grad again. Based on prev. It should not accumilate from prev backwards propogation 
        loss.backward()

    #update
    # We want to minize loss. Hvis hældningen af weighten negativ og vi skal have loss til at være mindre, skal weight.data være større. Når man går hen bliver f(x) mindre
    learning_rate = 1.0 - 0.99*k/100 # gradually decresing
    for p in model.parameters():
        p.data -= learning_rate * p.grad

    print(k, loss.data)

#print(loss)
aWeight = model.layers[0].neurons[0].w[0]
print(aWeight.grad)
#print(mlp.parameters())


