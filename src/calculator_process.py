import pandas as pd
import numpy as np

from concurrent import futures
import argparse
import grpc
import util
import time
import parameters_pb2_grpc
from worker import Worker
from lr import LogisticRegression


class Calculator(Worker):
    def __init__(self, epsilon, max_epoch):
        self._epsilon = epsilon
        self._max_epoch = max_epoch
        print('calculator init')
        super(Calculator, self).__init__()

    # return (param_list, delta)
    def algorithm(self, param_list, index):
        self.lr.update_w(param_list)
        batch_size = self.lr.batch_size
        i = index % ((self._data_end - self._data_begin)//batch_size)
        # print("minibatch ", i)
        real_begin = (i * batch_size) + self._data_begin
        real_end = ((i+1) * batch_size) + self._data_begin
        X = self.X[real_begin:real_end]
        y = self.y[real_begin:real_end]
        return self.lr.fit_one_batch(X, y, index)

    def calculator(self):
        print('enter calculator')
        self.X, self.y = self.read_data()
        self.lr = LogisticRegression(n_iter=70, eta=0.005, batch_size=10, gammar=0.5)
        while not self._begin_cal:
            time.sleep(1)
        start_time = time.time()
        st_clk = time.clock()
        print('start calculator')
        param_list = self.GetParams()
        param_list, loss = self.algorithm(param_list, 0)
        # index_epoch = (self._data_end - self._data_begin) // self.lr.batch_size
        index = 0
        while index < self._max_epoch:
            index += 1
            self.SetParams(param_list)
            param_list = self.GetParams()
            param_list, loss = self.algorithm(param_list, index)
            if index % 2000 == 0:
                print("Calculator.calculator::", index)
                print(loss)
        print('stop calculator', time.time() - start_time)
        print('CPU time', time.clock() - st_clk)

    def read_data(self):
        train = pd.read_csv(
            '../data/census-income.data.clean',
            header=None,
            delim_whitespace=True)
        train_y = train.values[:, -1]
        X = train.values[:, :-1]
        X = np.array(X, dtype='float64')
        train_X = np.copy(X)
        normal_list = [0, 126, 210, 211, 212, 353, 499]
        for i in normal_list:
            train_X[:, i] = (X[:, i] - X[:, i].mean()) / (X[:, i].std())
        return train_X, train_y


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--ip", default="localhost",
                        help="ip address")
    parser.add_argument("--port", default=9020, type=int,
                        help="ip port")
    parser.add_argument("--mip", default="localhost",
                        help="ip address")
    parser.add_argument("--mport", default=9000, type=int,
                        help="ip port")
    flags = parser.parse_args()
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
    process = Calculator(epsilon=0.1, max_epoch=50000)
    util.AddWorker(flags.mip, flags.mport, flags.ip, flags.port)
    parameters_pb2_grpc.add_WorkerServicer_to_server(process, server)
    server.add_insecure_port('{}:{}'.format(flags.ip, flags.port))
    server.start()
    process.calculator()
    util.FinishJob(flags.mip, flags.mport, flags.ip, flags.port)
    try:
        while(1):
            time.sleep(util._ONE_DAY_IN_SECONDS)
    except KeyboardInterrupt:
        server.stop(0)
