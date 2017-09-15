import numpy as np
np.random.seed(1)

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


def batch_iter(x, batch_size):
    n_samples = x.shape[0]
    start = 0
    end = batch_size
    while start < n_samples:
        yield x[start:end, ...]
        start, end = start + batch_size, end + batch_size

def train_test(model, X, Y, train_pct=.8, batch_size=1, learning_rate=1, epochs=10):
    n_samples = X.shape[0]
    losses = []
    test_losses = []
    
    X_train = X[:int(n_samples * train_pct), ...]
    Y_train = Y[:int(n_samples * train_pct), ...]
    X_test = X[int(n_samples * train_pct): , ...]
    Y_test = Y[int(n_samples * train_pct): , ...]

    # add the batch size in this function rather than delegate it to train
    for _ in range(epochs):
        for X_tr_b, Y_tr_b in zip(batch_iter(X_train, batch_size),
                                  batch_iter(Y_train, batch_size)):
            model.train(X_tr_b, Y_tr_b, learning_rate=learning_rate)
            # train loss:
            yhat_train = model.predict(X_train)
            losses.append(mse(yhat_train, Y_train))

            yhat_test = model.predict(X_test)
            test_losses.append(mse(yhat_test, Y_test))


    return losses, test_losses


    
    

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
            self.layers.append(np.random.randn(i, o)*0.02)
            
        self.activation = activation
        self.loss = loss

    def train(self, X, Y, learning_rate=.01, n_train=1, batch_size=None):
        """
        Train the network.
        X is a (number of examples x dimensionality of examples) matrix
        Y is a (number of examples x dimensionality of output) matrix

        n_train is the number of iterations to train

        returns a list of losses per iteration
        """
        if len(Y.shape) == 1: Y = Y.reshape(-1, 1)
        assert self.layers[0].shape[0] == X.shape[1]
        assert self.layers[-1].shape[1] == Y.shape[1]
        assert X.shape[0] == Y.shape[0]

        layer_outputs = []
        losses = []

        if not batch_size: batch_size = X.shape[0]

        for _ in range(n_train):
            for X_batch, Y_batch in zip(batch_iter(X, batch_size),
                                        batch_iter(Y, batch_size)):
                # forward pass.
                inp = X_batch
                for layer in self.layers:
                    assert inp.shape[1] == layer.shape[0]
                    out = self.activation(np.dot(inp, layer)) 
                    layer_outputs.append(out)
                    inp = out

                score = self.loss(out, Y_batch)
                losses.append(score)
                dscore = -self.loss(out, Y_batch, deriv=True)

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
                layer += learning_rate * X_batch.T.dot(delta)


            if (_+1) % (n_train/10) ==0:
                print('epoch {} of {}, loss: {:.5f}'.format(_+1,
                                                            n_train,
                                                            score))

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


            
                
            
            
            

