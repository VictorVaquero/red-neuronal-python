import numpy as np
import random
import argparse

import mnist_loader


DIF = 0.1


def sigmoid(v):
    return 1 / (1 + np.exp(-v))


def sigmoid_derivative(z):
    return sigmoid(z) * (1 - sigmoid(z))


def rectified_linear(v):
    return np.vstack([x if x > 0 else 0 for x in v])


def rectified_linear_derivative(z):
    return np.vstack([1 if x >= 0 else 0 for x in z])


def cost_derivative(c, v):
    return c - v


def equal(x, y):
    return (abs(x - y) < DIF).all()


class Red(object):
    """Red neuronal simple,
    cada columna representaria un nodo en cada matriz"""

    function_dictionary = {
        "sigmoid": sigmoid,
        "rectified": rectified_linear
    }
    function_derivatives_dictionary = {
        "sigmoid": sigmoid_derivative,
        "rectified": rectified_linear_derivative
    }

    def __init__(self, sizes, function):
        self.sizes = sizes
        self.layers = len(sizes)
        self.bias = [np.random.randn(i, 1) for i in sizes[1:]]
        self.weights = [np.random.randn(j, i)
                        for i, j in zip(sizes[:-1], sizes[1:])]
        self.function = function

    def feedforward(self, inp):
        for w, b in zip(self.weights, self.bias):
            inp = self.function_dictionary[self.function](np.dot(w, inp) + b)
        return inp

    def printf(self):
        print("\n\nWeights: {0}".format(self.weights))
        print("\nBias: {0}".format(self.bias))
    # Datos, size de cada entrenamiento por ciclo, numero ciclos,
    #  eta -> valor de actualizacion

    def train(self, data, mini_batch_size, epoch, eta, test=False):
        n = len(data)
        total = "Null"
        for x in range(epoch):
            random.shuffle(data)
            mini_data = [data[k:k + mini_batch_size]
                         for k in range(0, n, mini_batch_size)]
            for mini_batch in mini_data:
                self.update(mini_batch, eta)
            if test == True:
                total = self.evaluate(data)
            print(
                "Numero finalizado {0}, numero total: {1} de {2}".format(x, total, n))

    def update(self, data, eta):
        err_w = [np.zeros(w.shape) for w in self.weights]
        err_b = [np.zeros(b.shape) for b in self.bias]
        for x, y in data:
            delta_ew, delta_eb = self.backprop(x, y)
            err_w = [de + e for de, e in zip(delta_ew, err_w)]
            err_b = [de + e for de, e in zip(delta_eb, err_b)]
        self.weights = [w - e * eta / len(data)
                        for w, e in zip(self.weights, err_w)]
        self.bias = [b - e * eta / len(data) for b, e in zip(self.bias, err_b)]

    def backprop(self, inp, out):
        err_w = [np.zeros(w.shape) for w in self.weights]
        err_b = [np.zeros(b.shape) for b in self.bias]
        outs = []  # Value without sigmoid
        activation = inp
        activs = [inp]  # Value with sigmoid
        # Fordward
        for w, b in zip(self.weights, self.bias):
            activation = np.dot(w, activation) + b
            outs.append(activation)
            activation = self.function_dictionary[self.function](activation)
            activs.append(activation)
        # Backward
        # Delta es el error por nodo
        delta = cost_derivative(
            activs[-1], out) * self.function_derivatives_dictionary[self.function](outs[-1])
        err_b[-1] = delta
        # Vector columna por vector fila, crea matriz con errores
        err_w[-1] = np.dot(delta, activs[-2].transpose())
        for l in range(2, self.layers):
            delta = np.dot(self.weights[-l + 1].transpose(),
                           delta) * self.function_derivatives_dictionary[self.function](outs[-l])
            err_b[-l] = delta
            err_w[-l] = np.dot(delta, activs[-l - 1].transpose())
        return (err_w, err_b)

    def evaluate(self, test_data):
        #result = [(self.feedforward(x), y) for (x, y) in test_data]
        # return sum(int(equal(x, y)) for x, y in result)
        # Con np.argmax se hace una softmax
        result = [(np.argmax(self.feedforward(x)), np.argmax(y))
                  for (x, y) in test_data]
        return sum(int(i == j) for (i, j) in result)


# ------------------------ Main ---------------


parser = argparse.ArgumentParser(description="Simple red neuronal")
parser.add_argument("batch", type=int, help="Tamano de los mini lotes ")
parser.add_argument("epoch", type=int, help="Numero de ciclos")
parser.add_argument("eta", type=float, help="Valor aprendizaje")
parser.add_argument("-f", "--function", action="store", dest="function",
                    help="funcion a usar", default="rectified")
parser.add_argument("-t", "--test", action="store_true",
                    help="Calcular o no aciertos por ciclo")

args = parser.parse_args()


###################################################################################

training_data, validation_data, test_data = mnist_loader.load_data_wrapper()
training_data, validation_data, test_data = list(
    training_data), list(validation_data), list(test_data)
size = [784, 30, 10]


#---------------------------------------------------------------- XOR
# size = [2, 2, 1]
# data = [(np.array([[1], [1]]), 0), (np.array([[1], [0]]), 1),
#        (np.array([[0], [1]]), 1), (np.array([[0], [0]]), 0)]
#---------------------------------------------------------------- Binary
#
# n = 8
# size = [3, 3, 3]
# data = [(x, (x + 1) % n) for x in range(n)]
# data = [(list("{0:03b}".format(x)), list("{0:03b}".format(y)))
#        for x, y in data]
# data = [([int(x) for x in xn], [int(y) for y in yn]) for xn, yn in data]
# data = [(np.array(x).reshape(len(x), 1), np.array(y).reshape(len(x), 1))
#        for x, y in data]
#


##############################################################################


red = Red(size, args.function)
# red.printf()
red.train(training_data, args.batch, args.epoch, args.eta, test=args.test)

print(red.evaluate(validation_data))
# print data[15]

print("result {0}".format(red.feedforward(test_data[1][0])))
red.printf()
