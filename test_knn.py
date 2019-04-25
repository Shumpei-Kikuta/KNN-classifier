import numpy as np
import pandas as pd
import argparse
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from knn import calculate_similar_vector, main
from collections import Counter

iris = load_iris()
X = pd.DataFrame(iris.data, columns=["a", "b", "c", "d"])
Y = iris.target

# import pdb; pdb.set_trace()
train_X, test_X, train_Y, test_Y = train_test_split(X, Y, test_size=0.3, random_state=0)
train_X = train_X.reset_index()
test_X = test_X.reset_index()

def test_calculate_similar_vector_shape():
    idx = np.random.choice(test_X["index"].values)
    vector = test_X[test_X["index"] == idx].values[0, 1:]
    K = 10
    assert(calculate_similar_vector(vector, train_X, train_Y, K).shape == (K, ))

def test_calculate_similar_vector_value1():
    idx = np.random.choice(test_X["index"].values)
    vector = test_X[test_X["index"] == idx].values[0, 1:]
    K = 10
    count_arrays = calculate_similar_vector(vector, train_X, train_Y, K)
    uniques, counts = np.unique(count_arrays, return_counts=True)
    for i, (key_, value_) in enumerate(zip(uniques, counts)):
        if i == 0:
            max_key = key_
            max_num = value_
        else:
            if max_num <= value_:
                max_key = key_
                max_num = value_

    assert(max_key == test_Y[test_X["index"] == idx][0])
    
def test_calculate_similar_vector_value2():
    idx = np.random.choice(test_X["index"].values)
    vector = test_X[test_X["index"] == idx].values[0, 1:]
    K = 10
    count_arrays = calculate_similar_vector(vector, train_X, train_Y, K)
    uniques, counts = np.unique(count_arrays, return_counts=True)
    for i, (key_, value_) in enumerate(zip(uniques, counts)):
        if i == 0:
            max_key = key_
            max_num = value_
        else:
            if max_num <= value_:
                max_key = key_
                max_num = value_

    assert(max_key == test_Y[test_X["index"] == idx][0])

def test_main_shape():
    K = 10
    assert(main(train_X, test_X, train_Y, K).shape == test_Y.shape)
    
def test_main_value():
    K = 3
    rate = np.sum(main(train_X, test_X, train_Y, K) == test_Y) / test_Y.shape[0]
    assert(rate > 0.9)

def test_main_value2():
    K = 3
    rate = np.sum(main(train_X, test_X, train_Y, K) == test_Y) / test_Y.shape[0]
    assert(rate < 1)

if __name__ == '__main__':
    test_calculate_similar_vector_value1()