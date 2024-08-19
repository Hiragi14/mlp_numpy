import numpy as np

# 全結合層
class Dense:
    def __init__(self, in_dim, out_dim, function, deriv_function) -> None:
        # self.W = np.random.uniform(low=-0.08, high=0.08, size=(in_dim, out_dim)).astype("float64")
        self.W = np.zeros((in_dim, out_dim)).astype('float64')
        self.b = np.zeros(out_dim).astype("float64")

        self.function = function
        self.deriv_function = deriv_function

        self.x = None
        self.u = None
        self.dW = None
        self.db = None

        self.params_idxs = np.cumsum([self.W.size, self.b.size])
    def __call__(self, x):
        self.x = x
        self.u = np.matmul(self.x, self.W) + self.b
        h = self.function(self.u)
        return h
    def b_prop(self, delta, W):
        self.delta = self.deriv_function(self.u) * np.matmul(delta, W.T)
        return self.delta
    def compute_grad(self) -> None:
        batch_size = self.delta.shape[0]
        self.dW = np.matmul(self.x.T, self.delta) / batch_size
        self.db = np.matmul(np.ones(batch_size), self.delta) / batch_size