# -*- coding: utf-8 -*-
import nltk
import string
import numpy as np
import os
import random


en_stopwords = nltk.corpus.stopwords.words('english')
en_stopwords.extend(string.punctuation)
en_stopwords.extend(list(set(list(string.ascii_letters.lower()))))
en_stopwords.extend(["'s ", "``"])


def cleanText(text):
    tokenizer = nltk.word_tokenize(text.lower(), 'english')
    tokenText = list(set(tokenizer)-set(en_stopwords))
    return tokenText


# 1. clean and vectorize an input text file
# text: each file in the training dataset folder as a String
# dim: dimention of the glove model 50 / 100 / 150
def docToAvgGloveVec(text, gloveVec, dim):
    result = np.zeros((1, int(dim)))
    wc = 0
    text_list = cleanText(text)
    for word in text_list:
        # !! skip the word if it is does not exist in the glove_data_set
        if gloveVec.has_key(word):
            wc += 1
            result[0] += gloveVec[word]
    #2. return average of vector values of words in the document
    return result[0]/wc


# extract folder names in the dataset
# use folder names as labels
# path to training dataset folder
def tagList(path):
    tagLists = {}
    index = 0
    # tag : name of each folder in the training dataset (our labels)
    for tag in os.listdir(path):
        tagLists[tag] = index
        index += 1
    # create tagLists as : {"folder_1":0, "folder_2":1, ...}
    # i.e  {"neg":0, "pos":1}
    return tagLists


def createTrainingSet(main_path, gloveVec, dim):
    dataset = []  # accumulate all dataset into this
    labels = tagList(main_path)  # use folder names as labels
    for path in os.listdir(main_path):  # for each folder in dataset
        each_folder = os.path.join(main_path, path)
        for doc in os.listdir(each_folder):  # for each file in folder
            if doc.endswith(".txt"):
                f = open(os.path.join(each_folder, doc), "r")
                text = f.read().decode("utf8")
                f.close()  # extract the text in the file
                # represent text as a glove vector
                doc2glove = docToAvgGloveVec(text, gloveVec, dim)
                # numerical value of the label is val, i.e 0 / 1 ...
                val = [labels[path.split("/")[-1]]]
                # result is glove vector + itS value, X + Y
                result = np.append(doc2glove, val)
                dataset.append(result)
    # data set is too homogeneous
    # it needs to be shuffled
    random.shuffle(dataset)
    lc = len(dataset)
    print "all of the data set contains %d elements" % lc   
    # split data for training and test
    # ! maybe split for cross-validation too (later)
    train = np.array(dataset[0:int(lc*0.8)])  # use %80 as train
    test = np.array(dataset[int(lc*0.8):lc])  # use the rest for testing
    dim = int(dim)  # dimention in the glove dataset
    tr_x = train[:, 0:dim]
    tr_y = train[:, dim]  # last one is label (Y)
    ts_x = test[:, 0:dim]  # all others (x1, x2 .. xDim)
    ts_y = test[:, dim]
    # return train_x, train_y, test_x, test_y
    return np.matrix(tr_x), np.matrix(tr_y).T, np.matrix(ts_x), np.matrix(ts_y).T
