import parameters_pb2
import parameters_pb2_grpc


class Server(parameters_pb2_grpc.ServerServicer):
    def __init__(self):
        self._params = []

    def GetParams(self, request, context):
        # print("Server::GetParams")
        return parameters_pb2.Parameters(params=self._params)

    def InitParams(self, request, context):
        self._params = request.params
        # print("InitParams::", self._params)
        return parameters_pb2.StateMessage(status='OK')

    def SetParams(self, request, context):
        assert len(self._params) != 0
        assert request.end - request.begin == len(request.thetas)
        # print("Server::SetParams update {} paramters".format(len(request.thetas)))
        for i in range(request.begin, request.end):
            self._params[i] = request.thetas[i - request.begin]
        return parameters_pb2.StateMessage(status='OK')
