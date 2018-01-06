import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from matplotlib.colors import ListedColormap
from sklearn.linear_model import LogisticRegression


def main():
    train = pd.read_csv(
        '/Users/thinginitself/Downloads/census/census-income.data.clean',
        header=None,
        delim_whitespace=True)
    test = pd.read_csv(
        '/Users/thinginitself/Downloads/census/census-income.test.clean',
        header=None,
        delim_whitespace=True)
    train_y = train.values[:, -1]
    X = train.values[:, :-1]
    X = np.array(X, dtype='float64')
    train_X = np.copy(X)
    normal_list = [0, 126, 210, 211, 212, 353, 499]
    for i in normal_list:
        train_X[:, i] = (X[:, i] - X[:, i].mean()) / (X[:, i].std())

    test_y = test.values[:, -1]
    aX = test.values[:, :-1]
    aX = np.array(aX, dtype='float64')
    test_X = np.copy(aX)
    normal_list = [0, 126, 210, 211, 212, 353, 499]
    for i in normal_list:
        test_X[:, i] = (aX[:, i] - aX[:, i].mean()) / (aX[:, i].std())

    clf = LogisticRegression(max_iter=20)
    clf.fit(train_X, train_y)
    x = clf.score(test_X, test_y)
    print(x)


if __name__ == '__main__':
    main()
