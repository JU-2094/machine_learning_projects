# -*- coding:utf-8 -*-

import numpy as np
import pandas as pd
import math as m

__author__ = "Josué Fabricio Urbina González"


def read_data(path):
    df = pd.read_csv(path, sep='\t', index_col=0)
    data = df.as_matrix()
    return data[:, :-1], data[:, -1]


def sigmoid(x):
    return 1/(1+m.exp(-x))


def d1_sigmoid(x):
    return (1-sigmoid(x))*sigmoid(x)


def hardlim(x):
    if x >= 0:
        return 1
    else:
        return 0


# se incluye el bias en la matriz de pesos
class ModelPerceptron:

    def __init__(self):
        self.W = None

    def fit(self, X, Y, learning_rate, epochs, x_valid, y_valid):
        R, D = X.shape

        # adding bias
        W = np.random.rand(1, D+1)
        x0 = np.ones((X.shape[0], 1))
        X = np.append(X, x0, axis=1)
        x0 = np.ones((x_valid.shape[0], 1))
        x_valid = np.append(x_valid, x0, axis=1)

        fun = np.vectorize(hardlim)
        tunning = []
        for _ in range(epochs):
            for __ in range(1000):
                end = True
                for i in range(len(X)):
                    x = np.array([X[i]])
                    Yhat = fun(np.transpose(np.dot(W, np.transpose(x))))

                    e = np.square(Y[i]-Yhat)
                    W_n = W - 2*learning_rate*np.dot(e, x)
                    if not np.allclose(W_n, W):
                        end = False
                        W = W_n
                if end:
                    break

            Ypredict = fun(np.transpose(np.dot(W, np.transpose(x_valid))))
            tp = np.sum(Ypredict == y_valid)
            accuracy = tp / len(y_valid)
            tunning.append([W, accuracy])

            W = np.random.randn(1, D + 1)

        r = max(tunning, key=lambda k: k[1])
        self.W = r[0]
        return r[1]

    def predict(self, X):
        fun = np.vectorize(hardlim)
        x0 = np.ones((X.shape[0], 1))
        X = np.append(X, x0, axis=1)
        return fun(np.transpose(np.dot(self.W, np.transpose(X))))

    def evaluate(self, x_test, y_test):
        y_predict = self.predict(x_test)
        tp = np.sum(y_predict == y_test)
        accuracy = tp / len(y_test)
        return accuracy


class ModelMulticapa:

    def __init__(self):
        self.W0 = None
        self.W1 = None

    def fit(self, units_hidden, learning_rate, x_train, y_train, x_valid, y_valid):

        # adding bias

        x0 = np.ones((x_train.shape[0], 1))
        x_train = np.append(x_train, x0, axis=1)
        x0 = np.ones((x_valid.shape[0], 1))
        x_valid = np.append(x_valid, x0, axis=1)

        W0 = np.random.rand(units_hidden, x_train.shape[1])
        W1 = np.random.rand(1, units_hidden)

        logsigmoid = np.vectorize(sigmoid)
        d1_logsig = np.vectorize(d1_sigmoid)
        hardl = np.vectorize(hardlim)

        for _ in range(1000):
            stable = True
            for i in range(len(x_train)):
                x = np.transpose(np.array([x_train[i]]))
                y = np.array([y_train[i]])
                # forward

                a1 = logsigmoid(np.dot(W0, x))
                a2 = logsigmoid(np.dot(W1, a1))
                e = y-a2
                if m.fabs(e) > 0.01:
                    stable = False

                # backpropagation
                # sensibility
                s2 = -2 * e * d1_logsig(a2)

                fn = np.zeros((units_hidden, units_hidden))
                np.fill_diagonal(fn, d1_logsig(a1))

                s1 = fn.dot(np.transpose(W1)) * s2

                W1 = W1 - learning_rate * s2 * np.transpose(a1)
                W0 = W0 - learning_rate * s1 * np.transpose(x)

            if stable:
                break

        tmp = logsigmoid(np.dot(W0, np.transpose(x_valid)))
        y_predict = np.transpose(hardl(np.dot(W1, tmp)))

        tp = np.sum(y_predict == y_valid)
        accuracy = tp / len(y_valid)
        print('accuracy validation=',accuracy)
        self.W0 = W0
        self.W1 = W1

    def predict(self, X):
        logsigmoid = np.vectorize(sigmoid)
        hardl = np.vectorize(hardlim)
        x0 = np.ones((X.shape[0], 1))
        X = np.append(X, x0, axis=1)
        tmp = logsigmoid(np.dot(self.W0, np.transpose(X)))
        y_predict = np.transpose(hardl(np.dot(self.W1, tmp)))
        return y_predict

    def evaluate(self, x_test, y_test):
        y_predict = self.predict(x_test)
        tp = np.sum(y_predict == y_test)
        accuracy = tp / len(y_test)
        return accuracy


def main():
    path_data = "Data/corpus5/dataset.txt"
    x, y = read_data(path_data)
    train = int(0.7*len(x))
    valid = int(0.2*len(x))
    test = int(0.1*len(x))
    y = np.expand_dims(y, axis=1)

    x_train = x[0:train, :]
    x_valid = x[train:train+valid, :]
    x_test = x[train+valid: train+valid+test, :]
    y_train = y[0:train, :]
    y_valid = y[train:train + valid, :]
    y_test = y[train + valid: train + valid + test, :]

    # perceptrón simple
    # Loss square error
    model = ModelPerceptron()
    learning_rate = 0.1
    epochs = 5

    model.fit(x_train, y_train, learning_rate, epochs, x_valid, y_valid)
    score = model.evaluate(x_test, y_test)
    print('accuracy, perceptrón', score)

    print('pesos con bias W=', model.W)

    # multicapa
    # Mejor solución: 2 capas, de 10 unidades a 1 unidad
    # Función de activación entrenamiento, log-sigmoid
    # Loss absolute error
    units = 10
    model2 = ModelMulticapa()

    model2.fit(units, learning_rate, x_train, y_train, x_valid, y_valid)

    score = model2.evaluate(x_test, y_test)
    print('accuracy, multicapa', score)

    print('pesos con bias W0=', model2.W0, '\nW1=', model2.W1)


if __name__ == "__main__":
    main()
