import random
from motor import Value
import json

class Module:

    def zero_grad(self):
        for p in self.parameters():
            p.grad = 0

    def parameters(self):
        return []

class Neuron(Module):

    def __init__(self, nin, nonlin=True): # nin is number of inputs to the neuron
        self.w = [Value(random.uniform(-1,1)) for _ in range(nin)] # Random weight between -1 and 1
        self.b = Value(random.uniform(-1,1)) # Random bias between -1 and 1
        self.nonlin = nonlin

    def __call__(self, x): # makes it possible to write object(x)
        act = sum((wi*xi for wi,xi in zip(self.w, x)), self.b) # Pairs weight and input. Mulitply all weights with all inputs. Sum them

        return act.relu() if self.nonlin else act # Activation function, relu

    def parameters(self):
        return self.w + [self.b]

    def __repr__(self):
        return f"{'ReLU' if self.nonlin else 'Linear'}Neuron({len(self.w)})"

class Layer(Module):

    def __init__(self, nin, nout, **kwargs): # nout is number of outputs, number of neurons in the layer. Ex 2 dimensional neuron with 3 in the layer Layer(2, 3)
        self.neurons = [Neuron(nin, **kwargs) for _ in range(nout)]

    def __call__(self, x): 
        out = [n(x) for n in self.neurons]
        return out[0] if len(out) == 1 else out 

    def parameters(self):
        return [p for n in self.neurons for p in n.parameters()]

    def __repr__(self):
        return f"Layer of [{', '.join(str(n) for n in self.neurons)}]"

class MLP(Module): # The model. A list of Layers with Neurons

    def __init__(self, nin, nouts): # Takes number of dimension and a list of layers with neurons
        sz = [nin] + nouts
        self.layers = [Layer(sz[i], sz[i+1], nonlin=i!=len(nouts)-1) for i in range(len(nouts))]

    # Save the model weights to a file.
    def save_weights(self, file_path):
        weights = [p.data for p in self.parameters()] # Finds each weight in the neural network
        with open(file_path, 'w') as f:
            json.dump(weights, f)

    #Load the model weights from a file.
    def load_weights(self, file_path):
        with open(file_path, 'r') as f:
            weights = json.load(f)

        for p, w in zip(self.parameters(), weights): # Matches each pair from json file and weight in nn
            p.data = w

    def __call__(self, x):
        for layer in self.layers:
            x = layer(x)
        return x
    
    def parameters(self): # Returns weight for all neurons. 
        return [p for layer in self.layers for p in layer.parameters()]

    def __repr__(self):
        return f"MLP of [{', '.join(str(layer) for layer in self.layers)}]"
    
