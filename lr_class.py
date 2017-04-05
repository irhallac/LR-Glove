# encoding: utf-8
import numpy as np
import scipy.optimize as optimizer
import time
import gloveOpr as my_glove
import dataPreparations as dp


class LogisticRegression:

    def __init__(self, data_path, dim):
        self.gloveVec = my_glove.getGloveVec(dim)
        # i couldn't break this line !
        self.Train_X, self.Train_y, self.Test_X, self.Test_y = dp.createTrainingSet(data_path, self.gloveVec, dim)

    def get_Train_X(self):
        return self.Train_X

    def get_Train_y(self):
        return self.Train_y

    def get_Test_X(self):
        return self.Test_X

    def get_Test_y(self):
        return self.Test_y

    def get_Glove_Vec(self):
        return self.gloveVec

    def loadData(self, path):
        print "Loading data"

    def sigmoid(self, z):
        return 1.0 / (1 + np.exp(-z))

    def costFunction(self, theta, X, y):
        m, n = X.shape
        theta = np.matrix(theta).T
        g_z = self.sigmoid(X * theta)
        H = np.multiply(-y, np.log(g_z))
        T = np.multiply((1 - y), np.log(1 - g_z))
        cost = (np.sum(H - T) / m)
        return cost

    def gradient(self, theta, X, y):
        nData = X.shape[0]
        error = self.sigmoid(X * np.matrix(theta).T) - y
        grad = (1. / nData) * (X.T * error)
        return grad

    def predict(self, theta, X):
        response = self.sigmoid(X * theta.T)
        return [1 if x >= 0.5 else 0 for x in response]

    def train(self):
        print "training the dataset . . . "
        beginT = time.time()
        (m, n) = self.Train_X.shape
        thetas = np.zeros(n)
        trained = optimizer.fmin_tnc(func=self.costFunction,
                                     x0=thetas,
                                     fprime=self.gradient,
                                     args=(self.Train_X, self.Train_y),
                                     disp=False)
        theta_min = np.matrix(trained[0])
        print 'Training successfully completed \
                in %f seconds!' % (time.time() - beginT)
        return theta_min

    def test(self, theta_min):
        print "testing  . . . "
        nLines = len(self.Test_y)
        print 'Testing %d rows' % (nLines)
        predictions = self.predict(theta_min, self.Test_X)
        correct = [1 if ((p == 1 and t == 1) or (p == 0 and t == 0))
                   else 0 for (p, t) in zip(predictions, self.Test_y)]
        accuracy = sum(map(int, correct)) % len(correct)
        print " - "*10 + "Program Completed ! " + " - "*10
        print 'Correctly predicted = {0} of '.format(accuracy) + '\
           %d test items ' % (nLines)
