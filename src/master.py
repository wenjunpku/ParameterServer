import parameters_pb2_grpc
import util
import parameters_pb2
import time


class Master(parameters_pb2_grpc.MasterServicer):
    def __init__(self):
        '''
        (ip, port) : state
        '''
        # (ip,port):index
        self._server_list = {}
        # (ip,port):status
        self._worker_list = {}
        self._min_worker = 0
        self._result = []
        self._running = False

    def AddServer(self, request, context):
        ip, port = request.ip, request.port
        self._server_list[(ip, port)] = len(self._server_list)
        return parameters_pb2.StateMessage(status='OK')

    def AddWorker(self, request, context):
        ip, port = request.ip, request.port
        if self._running:
            self._worker_list[(ip, port)] = 'READY'
            '''
            set_range_thread = threading.Thread(target=self.SetRange())
            set_range_thread.start()
            self._worker_list[(ip, port)] = 'START'
            # call start worker
            util.StartWork(ip, port)
            '''
        else:
            self._worker_list[(ip, port)] = 'READY'
        return parameters_pb2.StateMessage(status='OK')

    def FinishJob(self, request, context):
        ip, port = request.ip, request.port
        self._worker_list[(ip, port)] = 'FINISHED'
        print("FinishJob::", self._worker_list)
        self._running = False
        if 'START' not in self._worker_list.values():
            # call server get params
            temp_res = []
            for ip, port in self._params_server_map:
                grpc_call = util.GrpcCallServer(ip, port)
                params = grpc_call.GetParams()
                begin = self._params_server_map[(ip, port)][0]
                temp_res.append((params, begin))
            temp_res = [(param_feature.result().params, begin) for param_feature, begin in temp_res]
            temp_res = sorted(temp_res, key=lambda x: int(x[1]))
            final_res = []
            for temp in temp_res:
                final_res.extend(temp[0])
            self._final_params = final_res
        return parameters_pb2.StateMessage(status='OK')

    def UserInit(self, start_work_num, data_set_num, params_num):
        self._min_worker = start_work_num
        self._data_set_num = data_set_num
        self._params_num = params_num

    def WorkerIsReady(self):
        return self._min_worker <= len(self._worker_list)

    def IsFinished(self):
        return 'START' not in self._worker_list.values()

    def GetFinalParams(self):
        return self._final_params

    # returns list of tuple
    def _partition(self, total, part_num):
        one_part = total // part_num + 1
        final_res = []
        for i in range(part_num - 1):
            final_res.append((i * one_part, (i + 1) * one_part))
        final_res.append(((part_num - 1) * one_part, total))
        return final_res

    def SetRange(self):
        server_num = len(self._server_list)
        worker_num = len(self._worker_list)
        self._params_server_map = {}  # (ip, port) : (begin, end)
        self._params_worker_map = {}  # (ip, port) : (begin, end)
        self._data_worker_map = {}    # (ip, port) : (begin, end)
        params_server_list = self._partition(self._params_num, server_num)
        print(params_server_list)
        params_worker_list = self._partition(self._params_num, worker_num)
        print(params_worker_list)
        data_part_list = self._partition(self._data_set_num, worker_num)
        print(data_part_list)
        for (i, server_info) in enumerate(self._server_list.keys()):
            self._params_server_map[server_info] = params_server_list[i]
        for (i, worker_info) in enumerate(self._worker_list.keys()):
            self._data_worker_map[worker_info] = data_part_list[i]
            self._params_worker_map[worker_info] = params_worker_list[i]

        # call set data range
        # call set params range
        # call set params Location
        print(self._params_worker_map)
        print(self._params_server_map)
        print(self._data_worker_map)
        send_location_list = []
        for key in self._params_server_map:
            print(key)
            send_location_list.append(
                (key[0], key[1], self._params_server_map[key][0],
                 self._params_server_map[key][1]))
        for worker_info in self._worker_list.keys():
            ip, port = worker_info
            util.SetDataRange(
                ip, port,
                self._data_worker_map[worker_info][0],
                self._data_worker_map[worker_info][1])
            util.SetParamsRange(
                ip, port,
                self._params_worker_map[worker_info][0],
                self._params_worker_map[worker_info][1],
            )
            util.SetParamsLocation(ip, port, send_location_list)
        print('Set Worker Done')

    def InitCluster(self, params_list):
        self.SetRange()
        for ip, port in self._params_server_map:
            begin = self._params_server_map[(ip, port)][0]
            end = self._params_server_map[(ip, port)][1]
            util.InitParams(ip, port, params_list[begin:end])
        print('Init Server Done')

    def StartCluster(self):
        self._running = True
        for worker_info in self._worker_list:
            self._worker_list[worker_info] = 'START'
            # call start worker
            util.StartWork(worker_info[0], worker_info[1])
        print('Cluster Started')
        while 'READY' in self._worker_list.values() or 'START' in self._worker_list.values():
            if 'READY' in self._worker_list.values():
                self.SetRange()
                for worker_info in self._worker_list:
                    if self._worker_list[worker_info] == 'READY':
                        self._worker_list[worker_info] = 'START'
                        util.StartWork(worker_info[0], worker_info[1])
            time.sleep(1)
