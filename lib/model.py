import numpy as np
from lib.dense import Dense

# モデルの定義
class Model:
    def __init__(self, hidden_dims, activation_functions, deriv_functions) -> None:
        self.layers = []
        for i in range(len(hidden_dims) - 2):
            self.layers.append(Dense(hidden_dims[i], hidden_dims[i+1], activation_functions[i], deriv_functions[i]))
        self.layers.append(Dense(hidden_dims[-2], hidden_dims[-1], activation_functions[-1], deriv_functions[-1]))
    def __call__(self, x):
        return self.forward(x)
    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x
    def backward(self, delta) -> None:
        for i, layer in enumerate(self.layers[::-1]):
            if i == 0:
                layer.delta = delta
                layer.compute_grad()
            else:
                delta = layer.b_prop(delta, W)
                layer.compute_grad()
            
            W = layer.W
    def update(self, eps=0.01) -> None:
        for layer in self.layers:
            layer.W -= eps * layer.dW
            layer.b -= eps * layer.db