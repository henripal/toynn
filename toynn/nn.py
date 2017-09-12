import numpy as np

"""
HELPER FUNCTIONS
feel free to define your own activation and loss functions
todo: maybe make this derivable function thing a class of its own
"""

def sigmoid(x, deriv=False):
    """
    return the sigmoid activation function or its derivative
    """
    if deriv:
        return x*(1-x)
    return 1/(1+np.exp(-x))

def mse(y, target, deriv=False):
    """
    return the mse between y and target
    note that this is minibatch safe
    """
    if len(target.shape) == 1: target = target.reshape(-1, 1)
    if len(y.shape) == 1: y = y.reshape(-1, 1)
    if deriv:
        return (y - target)/len(target)
    return np.mean(np.multiply((y - target), (y-target)))

"""
NEURAL NETWORK
"""

class NN:
    def __init__(self,layers, activation, loss):
        """
        Initialize the network.
        layers is a list containing (i, o) tuples 
        where i is the input size of the layer and
        o is the output size of the layer

        activation is the nonlinearity in the network; it should
        support differentiation.
        """
        self.layers = []

        for (i, o) in layers:
            # maybe the init should be parametrized as well
            self.layers.append(np.random.randn(i, o)/10)
            
        self.activation = activation
        self.loss = loss

    def train(self, X, Y, learning_rate=.01, n_train=1000):
        """
        Train the network.
        X is a (number of examples x dimensionality of examples) matrix
        Y is a (number of examples x dimensionality of output) matrix

        n_train is the number of iterations to train
        """
        if len(Y.shape) == 1: Y = Y.reshape(-1, 1)
        assert self.layers[0].shape[0] == X.shape[1]
        assert self.layers[-1].shape[1] == Y.shape[1]
        assert X.shape[0] == Y.shape[0]

        layer_outputs = []

        for _ in range(n_train):
            # forward pass.
            inp = X
            for layer in self.layers:
                assert inp.shape[1] == layer.shape[0]
                out = self.activation(np.dot(inp, layer)) 
                layer_outputs.append(out)
                inp = out

            score = self.loss(out, Y)
            dscore = -self.loss(out, Y, deriv=True)

            assert dscore.shape == out.shape
            error = dscore

            # backprop
            for layer in self.layers[:0:-1]:
                delta = error * self.activation(layer_outputs.pop(), deriv=True)
                error = delta.dot(layer.T)
                layer += learning_rate * layer_outputs[-1].T.dot(delta)
                
            # have to do the last layer differently 
            layer = self.layers[0]  # to maintain consistent semantics
            delta = error * self.activation(layer_outputs.pop(), deriv=True)
            error = delta.dot(layer.T)
            layer += learning_rate * X.T.dot(delta)

                
            print('loss: {0:.5f}'.format(score))

    def predict(self, X):
        """
        return the Yhat for the given X
        """
        assert self.layers[0].shape[0] == X.shape[1]
        inp = X
        for layer in self.layers:
            out = self.activation(np.dot(inp, layer)) 
            inp = out

        return out


            
                
            
            
            
