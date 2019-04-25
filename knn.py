import numpy as np
import pandas as pd
import argparse
from sklearn.datasets import load_iris
from collections import Counter

"""
input: 
    train_X: pd.DataFrame(N1, d)
    test_X: pd.DataFrame(N2, d)
    train_Y: pd.DataFrame(N1, 1)
    k: int
output:
    test_label: np.ndarrray(N2, k)
"""

def calculate_similar_vector(vector: np.ndarray, train_X: pd.DataFrame, train_Y: np.ndarray,  K: int) -> np.ndarray:
    """
    input: vector (shape=(d, )), train_X (shape=(N1, d)), train_Y (shape=(N1, ))
    output: sim_df (shape=(K, ))
    """
    norms = np.linalg.norm(train_X.iloc[:, 1:], ord=2, axis=1)
    inner_product_df = pd.DataFrame({"inner_product": np.dot(train_X.iloc[:, 1:], vector) / (norms * np.linalg.norm(vector, ord=2)), "snode": train_X["index"], "label": train_Y})

    argmax_df = inner_product_df.sort_values(by="inner_product", ascending=False)
    labels = argmax_df["label"].values[:K]
    return labels
    

def main(train_X: pd.DataFrame, test_X: pd.DataFrame, train_Y: np.ndarray, K: int) -> np.ndarray:
    test_size = test_X.shape[0]
    train_X = train_X.reset_index()
    test_X = test_X.reset_index()
    each_sim_df = pd.DataFrame(columns=["node"])
    for i in range(1, K + 1):
        each_sim_df["label{}".format(i)] = 0
    for i in range(test_size):
        node = test_X.index[i]
        vector = test_X.iloc[i, :].values
        each_sim_df[node] = calculate_similar_vector(vector, train_X, train_Y, K)




if __name__ == '__main__':
    # parser = argparse.ArgumentParser()
    # parser.add_argument("--input", default="",
    #                 help=)
    # args = parser.parse_args()
    iris = load_iris()
    train_X = iris.data
    test_X = iris.data
    train_Y = iris.target
    k = 1
    main(train_X, test_X, train_Y, k)