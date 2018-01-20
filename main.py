import numpy as np
import random
import argparse
import sys

import mnist_loader


DIF = 0.1


def sigmoid(v):
    return 1 / (1 + np.exp(-v))


def sigmoid_derivative(z):
    return sigmoid(z) * (1 - sigmoid(z))


def softmax(v):
    shiftv = v - np.max(v)
    exp = np.exp(shiftv)
    # if(np.isnan(v).any() or np.isnan(shiftv).any()):
    #     print(v, "       ", shiftv)
    #     sys.exit(0)
    return exp / sum(exp)


def softmax_derivative(z):
    x = z.shape[0]
    r = (np.identity(x) - np.tile(z, (1, x)))
    print(r, "    ", z)
    aux = np.sum(np.multiply(r, z.reshape((1, x))), 0)
    return aux.reshape((x, 1))


def rectified_linear(v):
    return np.vstack([x if x > 0 else 0 for x in v])


def rectified_linear_derivative(z):
    return np.vstack([1 if x >= 0 else 0 for x in z])


def cost_derivative_quadratic(inp, out, real_out, func):
    return (out - real_out) * func(inp)


def cost_derivative_cross(inp, out, real_out, func):
    """ Solo si la funcion de salida es la softmax
    """
    return (out - real_out)


def equal(x, y):
    return (abs(x - y) < DIF).all()


class Red(object):
    """Red neuronal simple,
    cada columna representaria un nodo en cada matriz"""

    function_dictionary = {
        "sigmoid": sigmoid,
        "rectified": rectified_linear,
        "softmax": softmax
    }
    function_derivatives_dictionary = {
        "sigmoid": sigmoid_derivative,
        "rectified": rectified_linear_derivative,
        "softmax": softmax_derivative
    }
    cost_derivatives_dictionary = {
        "quadratic": cost_derivative_quadratic,
        "cross": cost_derivative_cross
    }

    def __init__(self, sizes, output_function, hidden_function, cost_function):
        self.sizes = sizes
        self.layers = len(sizes)

        self.hidden_function = hidden_function
        self.output_function = output_function
        self.cost_function = cost_function
        if(cost_function == "cross" and output_function != "softmax"):
            print("Error en las funciones de entrada")

    def initialize_small(self):
        self.bias = [np.random.randn(i, 1) / i for i in self.sizes[1:]]
        self.weights = [np.random.randn(j, i) / (j + i)
                        for i, j in zip(self.sizes[:-1], self.sizes[1:])]

    def initialize_small_positive(self):
        self.bias = [abs(np.random.randn(i, 1)) / i for i in self.sizes[1:]]
        self.weights = [np.random.randn(j, i) / (j + i)
                        for i, j in zip(self.sizes[:-1], self.sizes[1:])]

    def initialize_large_positive(self):
        self.bias = [abs(np.random.randn(i, 1)) for i in self.sizes[1:]]
        self.weights = [np.random.randn(j, i)
                        for i, j in zip(self.sizes[:-1], self.sizes[1:])]

    def feedforward(self, inp):
        for w, b in zip(self.weights[:-1], self.bias[:-1]):
            inp = self.function_dictionary[self.hidden_function](
                np.dot(w, inp) + b)
        inp = self.function_dictionary[self.output_function](
            np.dot(self.weights[-1], inp) + self.bias[-1])
        return inp

    def printf(self):
        print("\n\nWeights: {0}".format(self.weights))
        print("\nBias: {0}".format(self.bias))
    # Datos, size de cada entrenamiento por ciclo, numero ciclos,
    #  eta -> valor de actualizacion

    def train(self, data, mini_batch_size, epoch, eta, test=False, test_data=None):
        n = len(data)
        total = "Null"
        for x in range(epoch):
            random.shuffle(data)
            mini_data = [data[k:k + mini_batch_size]
                         for k in range(0, n, mini_batch_size)]
            for mini_batch in mini_data:
                self.update(mini_batch, eta)
            if test == True:
                total = self.evaluate(test_data)
            print(
                "Numero finalizado {0}, numero total: {1} de {2}".format(x, total, len(test_data)))

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
        outputs = []  # Value without activation function
        act = inp
        activations = [inp]  # Value with activation function
        # Fordward
        for w, b in zip(self.weights[:-1], self.bias[-1]):
            act = np.dot(w, act) + b
            outputs.append(act)
            act = self.function_dictionary[self.hidden_function](
                act)
            activations.append(act)

        act = np.dot(self.weights[-1], act) + \
            self.bias[-1]
        outputs.append(act)
        act = self.function_dictionary[self.output_function](
            act)
        activations.append(act)

        # Backward
        # Delta es el error por nodo
        delta = self.cost_derivatives_dictionary[self.cost_function](
            outputs[-1], activations[-1], out, self.function_derivatives_dictionary[self.output_function])
        err_b[-1] = delta
        # Vector columna por vector fila, crea matriz con errores
        err_w[-1] = np.dot(delta, activations[-2].transpose())
        for l in range(2, self.layers):
            delta = np.dot(self.weights[-l + 1].transpose(),
                           delta) * self.function_derivatives_dictionary[self.hidden_function](outputs[-l])
            err_b[-l] = delta
            err_w[-l] = np.dot(delta, activations[-l - 1].transpose())

        return (err_w, err_b)

    def evaluate(self, test_data):
        # result = [(self.feedforward(x), y) for (x, y) in test_data]
        # return sum(int(equal(x, y)) for x, y in result)

        result = [(np.argmax(self.feedforward(x)), y)
                  for (x, y) in test_data]
        return sum(int(i == j) for (i, j) in result)


# ------------------------ Main ---------------


parser = argparse.ArgumentParser(description="Simple red neuronal")
parser.add_argument("batch", type=int, help="Tamano de los mini lotes ")
parser.add_argument("epoch", type=int, help="Numero de ciclos")
parser.add_argument("eta", type=float, help="Valor aprendizaje")
parser.add_argument("-hf", "--hidden", action="store", dest="hidden_function",
                    help="funcion oculta activacion", default="rectified")
parser.add_argument("-of", "--output", action="store", dest="output_function",
                    help="funcion de salida", default="softmax")
parser.add_argument("-c", "--cost", action="store", dest="cost_function",
                    help="funcion de coste", default="cross")
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


red = Red(size, args.output_function, args.hidden_function, args.cost_function)
red.initialize_small_positive()
print("Acertastes {}, de {} ".format(
    red.evaluate(validation_data), len(validation_data)))
print("result {} de {}".format(red.feedforward(
    validation_data[1][0]), validation_data[1][1]))
# red.printf()
try:
    red.train(training_data, args.batch, args.epoch,
              args.eta, test=args.test, test_data=test_data)
except KeyboardInterrupt:
    print("Acertastes {}, de {} ".format(
        red.evaluate(validation_data), len(validation_data)))
    # print data[15]

    print("result {} de {}".format(red.feedforward(
        validation_data[1][0]), validation_data[1][1]))
    red.printf()

print("Acertastes {}, de {} ".format(
    red.evaluate(validation_data), len(validation_data)))
# print data[15]

print("result {} de {}".format(red.feedforward(
    validation_data[1][0]), validation_data[1][1]))
red.printf()


# Sigmoid : valores de entrenamiento mas grandes
# Relus : Pesos y tita mas pequeñas ( sino se van de madre)
