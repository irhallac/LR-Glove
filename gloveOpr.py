# -*- coding: utf-8 -*-
# encoding: utf-8
import os
import numpy as np


def getGloveVec(dim):
    MAIN_PATH = os.path.dirname(__file__)
    glove_folder = os.path.join(MAIN_PATH, "glove")
    glove_file = os.path.join(glove_folder, "glove.6B."+dim+"d.txt")
    return glove2Vec(glove_file)


def glove2Vec(file):
    with open(file, "rb") as lines:
        vec = {line.split()[0]: np.array(map(float, line.split()[1:]))
               for line in lines}
    return vec
