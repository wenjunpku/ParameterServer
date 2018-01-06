import parameters_pb2_grpc
import util
import parameters_pb2


class Worker(parameters_pb2_grpc.WorkerServicer):
    def __init__(self):
        self._data_begin = 0
        self._data_end = -1
        self._param_begin = 0
        self._param_end = -1
        # (ip, port) : (begin, end)
        self._param_location_dict = {}
        self._params = []
        self._begin_cal = False
        self._grpc_call = {}

    def SetDataRange(self, request, context):
        self._data_begin = request.begin
        self._data_end = request.end
        print(self._data_begin, self._data_end)
        return parameters_pb2.StateMessage(status='OK')

    def SetParamsRange(self, request, context):
        self._param_begin = request.begin
        self._param_end = request.end
        print(self._param_begin, self._param_end)
        return parameters_pb2.StateMessage(status='OK')

    def SetParamsLocation(self, request, context):
        for params_location in request.params_location_list:
            self._param_location_dict[(params_location.ip, params_location.port)] = (
                params_location.begin, params_location.end)
            self._grpc_call[
                (params_location.ip, params_location.port)
            ] = util.GrpcCallServer(params_location.ip, params_location.port)
        print(self._param_location_dict)
        return parameters_pb2.StateMessage(status='OK')

    def StartWork(self, request, context):
        self._begin_cal = True
        print(self._data_begin,
              self._data_end,
              self._param_begin,
              self._param_end,
              self._param_location_dict,
              self._params,
              self._begin_cal)
        return parameters_pb2.StateMessage(status='OK')

    def calculator(self):
        raise NotImplementedError('Method not implemented!')

    def SetParams(self, params_list):
        begin, end = self._param_begin, self._param_end
        for ip, port in self._param_location_dict:
            send_server_begin = 0
            send_server_end = -1
            send_worker_begin = 0
            send_worker_end = -1
            param_range = self._param_location_dict[(ip, port)]
            if begin >= param_range[0] and end <= param_range[1]:
                # all worker's params at this server
                send_server_begin = begin - param_range[0]
                send_server_end = end - param_range[0]
                send_worker_begin = begin
                send_worker_end = end
                # print("Worker::SetParams::ONE", ip, port, send_worker_begin, send_worker_end, send_server_begin, send_server_end)
            elif begin >= param_range[0] and begin < param_range[1] and end > param_range[1]:
                # range(begin, param_range[1]) at this server
                send_server_begin = begin - param_range[0]
                send_server_end = param_range[1] - param_range[0]
                send_worker_begin = begin
                send_worker_end = param_range[1]
                # print("Worker::SetParams::TWO", ip, port, send_worker_begin, send_worker_end, send_server_begin, send_server_end)
            elif begin < param_range[0] and end > param_range[0] and end <= param_range[1]:
                # range(param_range[0], end) at this server
                send_server_begin = 0
                send_server_end = end - param_range[0]
                send_worker_begin = param_range[0]
                send_worker_end = end
                # print("Worker::SetParams::THR", ip, port, send_worker_begin, send_worker_end, send_server_begin, send_server_end)
            elif begin < param_range[0] and end > param_range[1]:
                # worker's range(param[0], param[1]) at this server
                send_server_begin = 0
                send_server_end = param_range[1] - param_range[0]
                send_worker_begin = param_range[0]
                send_worker_end = param_range[1]
                # print("Worker::SetParams::FOR", ip, port, send_worker_begin, send_worker_end, send_server_begin, send_server_end)
            # call server set param
            if send_server_begin < send_server_end:
                thetas = params_list[send_worker_begin: send_worker_end]
                grpc_call = self._grpc_call[(ip, port)]
                grpc_call.SetParams(send_server_begin, send_server_end, thetas)
            # print("Worker::SetParams server-ip:{} server-port:{}".format(ip, port))

    def GetParams(self):
        temp_res = []
        for ip, port in self._param_location_dict:
            grpc_call = self._grpc_call[(ip, port)]
            params = grpc_call.GetParams()
            begin = self._param_location_dict[(ip, port)][0]
            temp_res.append((params, begin))
            # print("Worker::GetParams server-ip:{} server-port:{}".format(ip, port))
        temp_res = [(param_feature.result().params, begin) for param_feature, begin in temp_res]
        temp_res = sorted(temp_res, key=lambda x: int(x[1]))
        final_res = []
        for temp in temp_res:
            final_res.extend(temp[0])
        return final_res
