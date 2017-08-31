import numpy as np
import random


DIF = 0.1


def sigmoid(v):
    return 1/ (1+np.exp(-v))
def cost_derivative(c,v):
    return c-v
# No hay mucha razon, se podria hacer mejor
def sigmoid_derivative(z):
    return sigmoid(z)*(1-sigmoid(z))
def equal(x,y):
    return abs(x-y) < DIF


class Red(object):
    """Red neuronal simple,
    cada columna representaria un nodo en cada matriz"""
    def __init__(self, sizes):
        self.sizes = sizes
        self.layers = len(sizes)
        self.bias = [np.random.randn(i,1) for i in sizes[1:]]
        self.weights = [np.random.randn(j,i) for i,j in zip(sizes[:-1],sizes[1:])]

    def feedforward(self, inp):
        for w,b in zip(self.weights, self.bias):
            inp = sigmoid(np.dot(w,inp) + b)
        return inp

    def printf(self):
        print "Weights: {0}".format(self.weights)
        print "Bias: {0}".format(self.bias)


    #Datos, size de cada entrenamiento por ciclo, numero ciclos,
    #  eta -> valor de actualizacion
    def train(self, data, mini_batch_size, epoch, eta, test = False):
        n = len(data)
        for x in xrange(epoch):
            random.shuffle(data)
            mini_data = [data[k:k+mini_batch_size] for k in xrange(0,n,mini_batch_size)]
            for mini_batch in mini_data:
                self.update(mini_batch,eta)
            if test == True:
                self.evaluate(data)
            print "Numero finalizado {0}".format(x)


    def update(self, data, eta):
        err_w = [np.zeros(w.shape) for w in self.weights]
        err_b = [np.zeros(b.shape) for b in self.bias]

        for x,y in data:
            delta_ew,delta_eb = self.backprop(x,y)
            err_w = [de+e for de,e in zip(delta_ew,err_w)]
            err_b = [de+e for de,e in zip(delta_eb,err_b)]
        self.weights = [w-e*eta/len(data) for w,e in zip(self.weights,err_w)]
        self.bias = [b-e*eta/len(data) for b,e in zip(self.bias,err_b)]



    def backprop(self, inp, out):
        err_w = [np.zeros(w.shape) for w in self.weights]
        err_b = [np.zeros(b.shape) for b in self.bias]

        outs = [] # Value without sigmoid
        activation = inp
        activs = [inp] # Value with sigmoid

        # Fordward

        for w,b in zip(self.weights,self.bias):
            
            activation = np.dot(w,activation)+b
            outs.append(activation)
            activation = sigmoid(activation)
            activs.append(activation)

        # Backward
        # Delta es el error por nodo
        delta = cost_derivative(activs[-1],out)*sigmoid_derivative(outs[-1])
        err_b[-1] = delta
        # Vector columna por vector fila, crea matriz con errores
        err_w[-1] = np.dot(delta, activs[-2].transpose())

        for l in xrange(2,self.layers):
            delta = np.dot(self.weights[-l+1].transpose(),delta)*sigmoid_derivative(outs[-l])
            err_b[-l] = delta
            err_w[-l] = np.dot(delta, activs[-l-1].transpose())
        return (err_w,err_b)

    def evaluate(self,test_data):
        # Con np.argmax se hace una softmax
        result = [(self.feedforward(x),y) for (x,y) in test_data]
        #print result
        return sum(int(equal(x,y)) for x,y in result)




red = Red([2,4,1])
red.printf()

data = [(np.array([[1],[1]]),0),(np.array([[1],[0]]),1),(np.array([[0],[1]]),1),(np.array([[0],[0]]),0)]
#print red.evaluate(data)
print red.feedforward(data[0][0])

red.train(data, 1, 10000, 0.1, test = True)

print red.feedforward(data[0][0])
print red.feedforward(data[1][0])
print red.feedforward(data[2][0])
print red.feedforward(data[3][0])

print red.evaluate(data)

red.printf()
