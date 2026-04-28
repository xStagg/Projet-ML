import marimo

__generated_with = "0.23.3"
app = marimo.App()


@app.cell
def _():
    import numpy as np
    import matplotlib.pyplot as plt

    return (np,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    NN from scratch
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Neuron Object
    """)
    return


@app.cell
def _(np):
    class Neuron:
        def __init__(self, input_size):
            self.weights = np.random.rand(input_size)
            self.bias = np.random.rand(1)

        def sigmoid(self, x):
            return 1 / (1 + np.exp(-x))

        def forward(self, inputs):
            z = np.dot(self.weights, inputs) + self.bias
            return self.sigmoid(z)

    return (Neuron,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Layer Object
    """)
    return


@app.cell
def _(Neuron, np):
    class Layer:
        def __init__(self, input_size, output_size):
            self.neurons = [Neuron(input_size) for _ in range(output_size)]

        def forward(self, inputs):
            return np.array([neuron.forward(inputs) for neuron in self.neurons])

    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Neural Network Object
    """)
    return


@app.cell
def _(np):
    class NeuralNetwork:
        def __init__(self, input_size, hidden_layers_size, output_size):
            self.input_size = input_size
            self.hidden_layers_size = hidden_layers_size
            self.output_size = output_size
        
            # Initialize weights
            self.W1 = np.random.rand(self.input_size, self.hidden_layers_size[0])
            self.W_hidden = []
            for i in range(1, len(self.hidden_layers_size)-1):
                self.W_hidden.append(np.random.rand(self.hidden_layers_size[i], self.hidden_layers_size[i+1]))
            self.W2 = np.random.rand(self.hidden_layers_size[-1], self.output_size)

        def Relu(self, x):
            return np.maximum(0, x)
        
        def sigmoid(self, x):
            return 1 / (1 + np.exp(-x))
    
        def sigmoid_derivative(self, x):
            return x * (1 - x)
    
        def forward(self, X):
            self.z1 = np.dot(X, self.W1)
            self.a1 = self.sigmoid(self.z1)
            self.z2 = np.dot(self.a1, self.W2)
            self.a2 = self.sigmoid(self.z2)
            return self.a2
    
        def backward(self, X, y, output):
            output_error = y - output
            output_delta = output_error * self.sigmoid_derivative(output)
        
            hidden_error = output_delta.dot(self.W2.T)
            hidden_delta = hidden_error * self.sigmoid_derivative(self.a1)
        
            # Update weights
            self.W2 += self.a1.T.dot(output_delta)
            self.W1 += X.T.dot(hidden_delta)

    return


if __name__ == "__main__":
    app.run()
