import numpy as np

def softmax(x):
    e = np.exp(x - np.max(x))
    e /= np.sum(e)
    return e


class sigmoid(object):
    @staticmethod
    def activate(x):
        return 1. / (1 + np.exp(-x))

    @staticmethod
    def derivative(a):
        return a * (1. - a)

    @staticmethod
    def get_name():
        return 'sigmoid'


class tanh(object):
    @staticmethod
    def activate(x):
        return (np.exp(x) - np.exp(-x)) / (np.exp(x) + np.exp(-x))

    @staticmethod
    def derivative(a):
        return 1. - (a * a)

    @staticmethod
    def get_name():
        return 'tanh'


class relu(object):
    @staticmethod
    def activate(x):
        return x * (x > 0)

    @staticmethod
    def derivative(a):
        return 1. * (a > 0)

    @staticmethod
    def get_name():
        return 'rectified linear'


if __name__ == '__main__':
    x = np.random.uniform(low=-0.001, high=0.001, size=(100,10))
    a = relu.activate(x)
    d = relu.derivative(a)