# Toy NN
A minimal, but OO and extensible neural network in python/numpy.
See [this notebook](./notebooks/usage.ipynb) for usage and an example.

## Implemented so far, and possible easy extensions:
- Batch GD only; should be easy to add Mini Batch GD and stochastic GD
- Activation function and loss functions could be reimplemented as part of
  an `autodiff` class rather than the ad hoc implementation of differentiation;
for now only sigmoid and MSE are implemented
- Only fully connected layers; convolutional layers should be easy to add;
  would require creating a `layer` class.

## Installation
I assume if you're using this you're going to want to modify it; so these
install instructions will let you modify the files any time, and the next
`import` will reflect those changes.

```bash
git clone https://github.com/henripal/toynn.git
cd toynn
pip install -e .
```

Now `import toynn` will work from anywhere.

See [notebook](./notebooks/usage.ipynb) for next steps!

