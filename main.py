#!/usr/bin/env python2
# -*- coding: utf-8 -*-

import lr_class as lr
import os
import pickle
import matplotlib.pyplot as plt
import collections

#def main(argv):


def plotLearningCurve(data, dim):
    
    X = data.keys()
    y = data.values()
    plt.figure(dim)
    plt.plot(X, y, linewidth=3, c='r')
    plt.ylim([(min(y)-10), (max(y)+10)])
    plt.xlim([(min(X)-10), (max(X)+10)])
    plt.xlabel('Training Data Size')
    plt.ylabel('Accuracy')
    plt.title('Learning rate for Glove dim: {}'.format(dim))
    plt.grid(True)
    plt.show()


def createLearningCurve(path, dimSizes):
    for dim in dimSizes:
        sizeAccAvr = {}
        # dataset is shuffled everytime
        # i think it is a good idea to do the tests for 10 times
        # and use the average accuracy as resulting learning rate
        for i in range(10):
            lr_c = lr.LogisticRegression(data_path=path, dim=dim)
            sizeAcc = lr_c.learningCurve()
            if len(sizeAccAvr.keys()) is 0:
                for key in sizeAcc.keys():
                    sizeAccAvr[key] = sizeAcc[key]
            else:
                for key in sizeAcc.keys():
                    sizeAccAvr[key] += sizeAcc[key]
        for key in sizeAccAvr.keys():
            sizeAccAvr[key] = sizeAccAvr[key]/10
        # serialize learning rates for each glove vector dimention-
        # using pickle
        fw = open(os.path.join("results",
                               "sizeAccAvr_{}.pkl".format(dim)), "w")
        pickle.dump(sizeAccAvr, fw)
        fw.close()


def loadLearningCurve(path, dimSizes):
    # plot all of the learning curves that are saved with pickle
    for dim in dimSizes:
        fr = open(os.path.join(path, "sizeAccAvr_{}.pkl".format(dim)), "r")
        sizeAccc = pickle.load(fr)
        fr.close()
        od = collections.OrderedDict(sorted(sizeAccc.items()))
        plotLearningCurve(od, dim)


def main():
    mainPath = os.getcwd()
    dataPath = os.path.join(mainPath, "review_polarity")
    resultPath = os.path.join("results")
    dimSizes = ["50", "100", "200", "300"]
    createLearningCurve(dataPath, dimSizes)
    loadLearningCurve(resultPath, dimSizes)

if __name__ == "__main__":
    main()
    #main(sys.argv[1:])
