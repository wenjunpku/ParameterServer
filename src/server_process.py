from concurrent import futures

import argparse
import time
import grpc
import util
import parameters_pb2_grpc
from server import Server


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--ip", default="localhost",
                        help="ip address")
    parser.add_argument("--port", default=9010, type=int,
                        help="ip port")
    parser.add_argument("--mip", default="localhost",
                        help="ip address")
    parser.add_argument("--mport", default=9000, type=int,
                        help="ip port")
    flags = parser.parse_args()
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
    process = Server()
    util.AddServer(flags.mip, flags.mport, flags.ip, flags.port)
    parameters_pb2_grpc.add_ServerServicer_to_server(process, server)
    server.add_insecure_port('{}:{}'.format(flags.ip, flags.port))
    server.start()
    try:
        while(1):
            time.sleep(util._ONE_DAY_IN_SECONDS)
    except KeyboardInterrupt:
        server.stop(0)
