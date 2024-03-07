import math
import numpy as np
import matplotlib.pyplot as plt
from graphviz import Digraph

def f(x):
    return 3*x**2 - 4*x + 5
# We want to find derivates of the different x. We dont actually find the derivate for the nn. That would be a looong expression.
# We just need to know that what a derivative is. h -> 0. Hældningen. Hvis vi gør x en smule større, hvordan reagerer funktionen så? Større eller mindre.   
h = 0.0001
x = 3.0
haeldningen = (f(x + h) - f(x))/h # Hvor meget funktionen reagerer, negativt eller positivt. Og så divideret med h for at normalisere
#print(haeldningen)

#print(xs)
# We make a value class to make a value object
class Value: 
    def __init__(self, data, _children=(), _op=''): # Children are the values creating the data. What created the value. op is how the value was created
        self.data = data
        self.grad = 0 # No effect of the output at init
        self._backward = lambda: None # Function to calculate chain rule. Propogate gradient. Does not do anything by defualt because of leaf node. Should not calculate derivative because its the last. 
        self._prev = set(_children)
        self._op = _op # Operation that produced this node

    def __repr__(self): # How it shows when we print
        return f"Value(data={self.data})"
    
    def __add__(self, other): # When using '+'. Adding two value objects together. a + b = a.__add__b
        other = other if isinstance(other, Value) else Value(other) # To make it possible to write Value(x) + y. Makes int to value object if not already
        out = Value(self.data + other.data, (self, other), '+')
        def _backward(): # How gradient is calculated 
            # Both nodes gets a gradient
            self.grad += 1.0 * out.grad # local derivative is 1 because we just add. Therefore derivative of ydre(indre) * 1 = ydre(indre), som er grad. Chain rule.
            other.grad += 1.0 * out.grad
        out._backward = _backward # dont call just store function
        return out
    
    def __mul__(self, other): # When using '*'. Mulitplying two value objects together
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data * other.data, (self, other), '*')
        def _backward(): 
            self.grad += other.data * out.grad
            other.grad += self.data * out.grad
        out._backward = _backward
        return out
    
    def tanh(self): # Hyperbolic tanh function. Activation function
        n = self.data
        t = (math.exp(2*n)-1)/(math.exp(2*n)+1)
        out = Value(t, (self, ))
        def _backward():
            self.grad += (1 - t**2) * out.grad # 1-tanh(n)^2 * outer grad. Last node.grad should be 1. Derivative of L to L is 1. 
            out._backward = _backward
        return out
    
    def __pow__(self, other):
        assert isinstance(other, (int, float)), "only supporting int/float powers for now"
        out = Value(self.data**other, (self,), f'**{other}')

        def _backward():
            self.grad += (other * self.data**(other-1)) * out.grad
        out._backward = _backward

        return out

    def relu(self):
        out = Value(0 if self.data < 0 else self.data, (self,), 'ReLU')

        def _backward():
            self.grad += (out.data > 0) * out.grad
        out._backward = _backward

        return out
    
    # We would like to go through the tree topologically to backwards propogate.
    # topological order all of the children in the graph
    def backward(self):
        # topological order all of the children in the graph
        topo = []
        visited = set()
        def build_topo(v):
            if v not in visited:
                visited.add(v)
                for child in v._prev:
                    build_topo(child)
                topo.append(v)
        build_topo(self)

        # go one variable at a time and apply the chain rule to get its gradient
        self.grad = 1.0 # Start self to be 1. Top of the chain is 1 is derivative of itself (like f(x)=x) is 1
        for node in reversed(topo):
            node._backward()

    def __neg__(self): # -self
        return self * -1

    def __radd__(self, other): # other + self
        return self + other

    def __sub__(self, other): # self - other
        return self + (-other)

    def __rsub__(self, other): # other - self
        return other + (-self)

    def __rmul__(self, other): # other * self
        return self * other

    def __truediv__(self, other): # self / other
        return self * other**-1

    def __rtruediv__(self, other): # other / self
        return other * self**-1

    
x1 = Value(data=2.0)
x2 = Value(data=0.0)
w1 = Value(data=-3.0)
w2 = Value(data=1.0)
b = Value(data=6.8813735870195432)
x1w1 = x1*w1
x2w2 = x2*w2
x1w1x2w2 = x1w1 + x2w2
n = x1w1x2w2 + b
o = n.tanh()
o.backward()

