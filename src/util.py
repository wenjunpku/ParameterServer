import grpc
import parameters_pb2
import parameters_pb2_grpc


_ONE_DAY_IN_SECONDS = 60 * 60 * 24


# Worker GRPC
def SetDataRange(ip, port, begin, end):
    channel = grpc.insecure_channel('{}:{}'.format(ip, port))
    stub = parameters_pb2_grpc.WorkerStub(channel)
    stub.SetDataRange(parameters_pb2.Range(begin=begin, end=end))


def SetParamsRange(ip, port, begin, end):
    channel = grpc.insecure_channel('{}:{}'.format(ip, port))
    stub = parameters_pb2_grpc.WorkerStub(channel)
    stub.SetParamsRange(parameters_pb2.Range(begin=begin, end=end))


def SetParamsLocation(ip, port, send_location_list):
    channel = grpc.insecure_channel('{}:{}'.format(ip, port))
    stub = parameters_pb2_grpc.WorkerStub(channel)
    send_list = []
    for location in send_location_list:
        send_list.append(parameters_pb2.ParamsLocation(
            ip=location[0],
            port=location[1],
            begin=location[2],
            end=location[3]))
    stub.SetParamsLocation(
        parameters_pb2.ParamsLocations(params_location_list=send_list))


def StartWork(ip, port):
    channel = grpc.insecure_channel('{}:{}'.format(ip, port))
    stub = parameters_pb2_grpc.WorkerStub(channel)
    stub.StartWork(parameters_pb2.StateMessage(status='OK'))


class GrpcCallServer(object):

    def __init__(self, ip, port):
        self._ip = ip
        self._port = port
        channel = grpc.insecure_channel('{}:{}'.format(ip, port))
        self.stub = parameters_pb2_grpc.ServerStub(channel)

    # Server GRPC
    def GetParams(self):
        response_future = self.stub.GetParams.future(
            parameters_pb2.StateMessage(status='OK'))
        return response_future

    def SetParams(self, begin, end, thetas):
        self.stub.SetParams.future(parameters_pb2.RequestChange(begin=begin, end=end, thetas=thetas))
        # self.stub.SetParams(parameters_pb2.RequestChange(begin=begin, end=end, thetas=thetas))


def InitParams(ip, port, params_list):
    channel = grpc.insecure_channel('{}:{}'.format(ip, port))
    stub = parameters_pb2_grpc.ServerStub(channel)
    stub.InitParams(parameters_pb2.Parameters(params=params_list))


#   Master GRPC
def AddWorker(ip, port, worker_ip, worker_port):
    channel = grpc.insecure_channel('{}:{}'.format(ip, port))
    stub = parameters_pb2_grpc.MasterStub(channel)
    stub.AddWorker(parameters_pb2.NodeInfo(ip=worker_ip, port=worker_port))


def AddServer(ip, port, server_ip, server_port):
    channel = grpc.insecure_channel('{}:{}'.format(ip, port))
    stub = parameters_pb2_grpc.MasterStub(channel)
    stub.AddServer(parameters_pb2.NodeInfo(ip=server_ip, port=server_port))


def FinishJob(ip, port, worker_ip, worker_port):
    channel = grpc.insecure_channel('{}:{}'.format(ip, port))
    stub = parameters_pb2_grpc.MasterStub(channel)
    stub.FinishJob(parameters_pb2.NodeInfo(ip=worker_ip, port=worker_port))
