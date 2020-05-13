import numpy as np
import numpy.random as random

class BP:

    def __init__(self, Numino, Numhno, Numono):
        self.Numino = Numino
        self.Numhno = Numhno
        self.Numono = Numono

        self.whi = None
        self.woh = None
        self.bh = None
        self.bo = None
        self.lr = 0

    def inital(self, lr):
        self.whi = random.normal(0, pow(self.Numhno, -0.5), [self.Numhno, self.Numino])
        self.woh = random.normal(0, pow(self.Numono, -0.5), [self.Numono, self.Numhno])

        self.bh = random.normal(0, pow(self.Numino, -0.5), [self.Numhno, 1])
        self.bo = random.normal(0, pow(self.Numhno, -0.5), [self.Numono, 1])
        self.lr = lr
        
    def f(self, x):
        return 1 / (1+np.exp(-x))

    def predict(self, X):
        X = X.reshape(self.Numino, 1)
        y_ = self.whi.dot(X)
        yh = self.f(y_+self.bh)
        z_ = self.woh.dot(yh)
        return self.f(z_+self.bo)

    def train(self, X, Y):
        X = X.reshape(self.Numino, 1)
        Y = Y.reshape(self.Numono, 1)
        y_ = self.whi.dot(X)
        y_ = self.whi.dot(X)
        yh = self.f(y_+self.bh)
        z_ = self.woh.dot(yh)
        z = self.f(z_+self.bo)

        loss = np.power(Y-z, 2)
        err_out = Y-z
        err_hid = self.woh.T.dot(err_out)

        self.woh += self.lr*(err_out*z*(1-z)).dot(yh.T)
        self.whi += self.lr*(err_hid*yh*(1-yh)).dot(X.T)

        self.bo += self.lr*err_out*z*(1-z)
        self.bh += self.lr*err_hid*yh*(1-yh)
        return self.whi, self.woh, self.bh, self.bo

