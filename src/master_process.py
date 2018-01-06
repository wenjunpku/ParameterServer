from concurrent import futures

import argparse
import grpc
import util
import time
import parameters_pb2_grpc
from master import Master
from lr import LogisticRegression
import pandas as pd
import numpy as np


def run(process):
    start_work_num = int(input("Please Input Minimal Worker Number:"))
    # data_set_num = int(input("Please Input Data Set Length:"))
    params_num = int(input("Please Input Parameters Length:"))

    data_set_num = 199520
    process.UserInit(start_work_num, data_set_num, params_num)
    while not process.WorkerIsReady():
        time.sleep(3)
    params_list = [0.0 for i in range(params_num)]
    process.InitCluster(params_list)
    process.StartCluster()
    while not process.IsFinished():
        time.sleep(3)
    print("FINAL RESULT!!", process.GetFinalParams())
    infer(process.GetFinalParams())


def infer(w):
    test = pd.read_csv(
        '../data/census-income.test.clean',
        header=None,
        delim_whitespace=True)
    test_y = test.values[:, -1]
    X = test.values[:, :-1]
    X = np.array(X, dtype='float64')
    test_X = np.copy(X)
    normal_list = [0, 126, 210, 211, 212, 353, 499]
    for i in normal_list:
        test_X[:, i] = (X[:, i] - X[:, i].mean()) / (X[:, i].std())
    lr = LogisticRegression()
    print(lr.score(test_X, test_y, w))
    lr.F1(test_X, test_y, w)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--ip", default="localhost",
                        help="ip address")
    parser.add_argument("--port", default=9000, type=int,
                        help="ip port")
    flags = parser.parse_args()
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
    process = Master()
    parameters_pb2_grpc.add_MasterServicer_to_server(process, server)
    server.add_insecure_port('{}:{}'.format(flags.ip, flags.port))
    server.start()
    try:
        run(process)
        while(1):
            time.sleep(util._ONE_DAY_IN_SECONDS)
    except KeyboardInterrupt:
        server.stop(0)
