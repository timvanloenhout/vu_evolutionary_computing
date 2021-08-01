from controller import Controller
import numpy as np
import sys


# Based on demo_controller.py

def sigmoid_activation(x):
    return 1. / (1. + np.exp(-x))

class Neuron_Controller(Controller):
    def __init__(self):
        pass

    def control(self, inputs, controller):
        # Normalise to unit length
        inputs /= np.linalg.norm(inputs)

        w1 = controller[:200].reshape((len(inputs), 10))
        w2 = controller[200:250].reshape(10, 5)
        b1 = controller[250:260].reshape(1, 10)
        b2 = controller[260:265].reshape(1, 5)

        h = np.tanh(inputs @ w1 + b1)
        output = sigmoid_activation(h @ w2 + b2)[0]

        # return decisions
        return np.rint(output)
