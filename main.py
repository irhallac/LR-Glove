#!/usr/bin/env python2
# -*- coding: utf-8 -*-

import lr_class as lr
import os


#def main(argv):
def main():
    mainPath = os.getcwd()
    dataPath = os.path.join(mainPath, "review_polarity")
    lr_c = lr.LogisticRegression(data_path=dataPath, dim="50")
    weights = lr_c.train()
    lr_c.test(weights)


if __name__ == "__main__":
    main()
    #main(sys.argv[1:])
