#from engine import Value
from engine import Value
import random
class Neuron:
    def __init__ (self, nin): # nin is number of inputs to the neuron
        self.w = [Value(random.uniform(-1,1)) for _ in range(nin)] # Creates weight between -1 and 1
        self.b = Value(random.uniform(-1,1)) # random bias between -1 and 1

    def __call__(self, x): # makes it possible to write object(x)
        # w * x + b
        # Mulitply all weights with all inputs
        act = sum(wi*xi for wi, xi in zip(self.w, x)) + self.b # Pairs two lists together. Mulitplies each pair, sums all pairs and adds b.
        out = act.tanh()
        return out
    
    def parameters(self):
        return self.w + [self.b]
    
x = [2.0, 3.0] 
n = Neuron(2)# init two dimensional neuron
#print(n(x))

# A layer of x neurons with y dimensions
class Layer: 
    def __init__(self, nin, nout): # nout is number of outputs, number of neurons. nin is number of dimensions
        self.neurons = [Neuron(nin) for _ in range(nout)] # Make nout neurons with each x number of inputs (nin). Ex: two dimensional neurons with 3 in the layer. Making 3 neurons. Layer(2, 3)

    def __call__(self, x): # Independently evaluate them 
        outs = [n(x) for n in self.neurons]
        return outs[0] if len(outs) == 1 else outs
    
    def parameters(self):
        return [p for neuron in self.neurons for p in neuron.parameters()]
    
class MLP: # A list of Layers with Neurons
    def __init__(self, nin, nouts): # Takes number of dimension and a list of layers with neurons
        sz = [nin] + nouts
        self.layers = [Layer(sz[i], sz[i+1]) for i in range(len(nouts))]

    def __call__(self, x):
        for layer in self.layers:
            x = layer(x)
        return x
    
    def parameters(self):
        return [p for layer in self.layers for p in layer.parameters()]
# We can make an input layer with 3 inputs. 2 hidden layers with 4 neurons and output layer with 1 neuron.
mlp = MLP(3, [4, 4, 1])
x = [2.0, 3.0, -1.0] # inputs
mlp(x)

# Four inputs
xs = [
    [2.0, 3.0, -1.0],
    [3.0, -1.0, 0.5],
    [0.5, 1.0, 1.0],
    [1.0, 1.0, -1.0],
]

# Based on first row input first row output should be 1. Second row -1 and so on.
ys = [1.0, -1.0, -1.0, 1.0] # goal

# Gradient descent
for k in range(10):
    # forward pass
    ypred = [mlp(x) for x in xs] # result after inputs and weights
    # We need to tune weights to achieve the goal ys
    loss = sum([(yout - ygt)**2 for ygt, yout in zip(ys, ypred)]) # measures how well the neural net is performing. Y ground truth. result - goal for each neuron. summed we get the loss 
    
    # backward pass
    for p in n.parameters():
        p.grad = 0.0 # Start at zero again. Because we start at the top of the tree and run down so every node will get a grad again. Based on prev. It should not accumilate from prev backwards propogation 
        loss.backward()

    #update
    # We want to minize loss. Hvis hældningen af weighten negativ og vi skal have loss til at være mindre, skal weight.data være større. Når man går hen bliver f(x) mindre
    for p in n.parameters():
        p.data += -0.01 * p.grad

    print(k, loss.data)


#print(loss)
aWeight = mlp.layers[0].neurons[0].w[0]
#print(aWeight.grad)
#print(mlp.parameters())



