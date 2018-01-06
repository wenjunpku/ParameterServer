import numpy as np
import math


class LogisticRegression(object):
    """LogisticRegression classifier.

    Parameters
    ------------
    eta : float
        Learning rate (between 0.0 and 1.0)
    n_iter : int
        Passes over the training dataset.
    gammar : float
        decal Learning rate's rate

    Attributes
    -----------
    w_ : 1d-array
        Weights after fitting.
    cost_ : list
        Cost in every epoch.

    """
    def __init__(self, eta=0.01, n_iter=50, batch_size=10, gammar=0.5):
        self.eta = eta
        self.n_iter = n_iter
        self.batch_size = batch_size
        self.gammar = gammar

    def update_w(self, param_list):
        self.w_ = np.array(param_list)

    def fit_one_batch(self, X_batch, y_batch, index):
        if X_batch.shape[0] == 0:
            return self.w_, 1.5
        self.cost_ = []
        if index != 0 and index % 100000 == 0:
            self.eta *= self.gammar
        y_val = self.activation(X_batch)
        loss = self._new_cost(X_batch, y_batch)
        errors = (y_batch - y_val)
        neg_grad = X_batch.T.dot(errors)
        self.w_[1:] += self.eta * neg_grad
        self.w_[0] += self.eta * errors.sum()
        return self.w_, loss

    def fit(self, X, y):
        """ Fit training data.

        Parameters
        ----------
        X : {array-like}, shape = [n_samples, n_features]
            Training vectors, where n_samples is the number of samples and
            n_features is the number of features.
        y : array-like, shape = [n_samples]
            Target values.
        alpha : float
            Regularization parameter.

        Returns
        -------
        self : object

        """
        self.w_ = np.zeros(1 + X.shape[1])
        self.cost_ = []
        batch_num = X.shape[0] // self.batch_size
        for i in range(self.n_iter):
            if i != 0 and i % 10 == 0:
                self.eta *= self.gammar
                print(self.eta)
            loss_iter = 0
            for j in range(batch_num):
                X_batch = X[j*self.batch_size:(j+1)*self.batch_size]
                if X_batch.shape[0] == 0:
                    break
                y_batch = y[j*self.batch_size:(j+1)*self.batch_size]

                y_val = self.activation(X_batch)
                loss = self._new_cost(X_batch, y_batch)
                # loss = self._new_logit_cost(y_batch, y_val)
                # print('iter', i, 'batch', j)
                # print('loss', loss)
                loss_iter += loss
                errors = (y_batch - y_val)
                neg_grad = X_batch.T.dot(errors)
                self.w_[1:] += self.eta * neg_grad
                self.w_[0] += self.eta * errors.sum()

            print('iter', i)
            print('loss_iter', loss_iter/batch_num)
            self.cost_.append(loss_iter)
        return self

    def _logit_cost(self, y, y_val):

        logit = -y.dot(np.log(y_val)) - ((1 - y).dot(np.log(1 - y_val)))
        return logit

    def _new_logit_cost(self, y, y_val):
        ans = 0
        for i in range(len(y)):
            if y[i] == 1:
                ans -= math.log(y_val[i])
            else:
                ans -= math.log(1-y_val[i])
        return ans

    def _new_cost(self, X, y):
        return -np.sum(self.new_activation(X, y))

    def _sigmoid(self, z):
        return (1.0 / (1.0 + np.exp(-z)))

    def net_input(self, X):
        """Calculate net input"""
        return np.dot(X, self.w_[1:]) + self.w_[0]

    def activation(self, X):
        """ Activate the logistic neuron"""
        z = self.net_input(X)
        return self._sigmoid(z)

    def new_activation(self, X, y):
        """ Activate the logistic neuron"""
        yz = y.dot(self.net_input(X))
        return np.log(self._sigmoid(yz))

    def predict_proba(self, X):
        """
        Predict class probabilities for X.

        Parameters
        ----------
        X : {array-like, sparse matrix}, shape = [n_samples, n_features]
            Training vectors, where n_samples is the number of samples and
            n_features is the number of features.

        Returns
        ----------
          Class 1 probability : float

        """
        return self.activation(X)

    def predict(self, X):
        """
        Predict class labels for X.

        Parameters
        ----------
        X : {array-like, sparse matrix}, shape = [n_samples, n_features]
            Training vectors, where n_samples is the number of samples and
            n_features is the number of features.

        Returns
        ----------
        class : int
            Predicted class label.

        """
        # equivalent to np.where(self.activation(X) >= 0.5, 1, 0)
        return np.where(self.net_input(X) >= 0.0, 1, 0)

    def score(self, X, y, w):
        self.w_ = np.array(w)
        y_predict = self.predict(X)
        rights = 0
        for k1, k2 in zip(y_predict, y):
            if k1 == k2:
                rights += 1
        print(rights)
        print(len(y))
        return (rights+0.0) / len(y)

    def F1(self, X, y, w):
        self.w_ = np.array(w)
        y_predict = self.predict(X)
        tp = 0
        fp = 0
        fn = 0
        tn = 50
        for k1, k2 in zip(y_predict, y):
            if k1 == 1 and k2 == 1:
                tp += 1
            elif k1 == 1 and k2 == 0:
                fp += 1
            elif k1 == 0 and k2 == 1:
                fn += 1
            elif k1 == 0 and k2 == 0:
                tn += 1
        precision = tp / (tp + fp)
        recall = tp / (tp + fn)
        f1 = (2 * precision * recall) / (precision + recall)
        print('precision', precision)
        print('recall', recall)
        print('F1', f1)
        return f1
