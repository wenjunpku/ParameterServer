# package request
- python -m pip install --upgrade pip
- python -m pip install grpcio
- python -m pip install grpcio-tools

# generate python class file by proto file
- cd python
- python -m grpc_tools.protoc -I../protos --python_out=. --grpc_python_out=. ../protos/parameters.proto

# Start Server
- cd python
- python server.py

# Start Worker
- cd python
- python worker.py


# Basic Design Rules
## Master
- Separate dataset by number of workers
- When all Worker finish the job, inform Server to save parameters to hard disk
- Then all Nodes stop operation
- When worker number changes, redecided data partition
- initial waiting time for register

## Server
- Provide PUSH and PULL method
- Wait? Lock?

## Worker
- Communicate With master to decide the data partition
- Every step finished, use push and pull to update parameters
- Report to Master when finish the job
- Inform Master when join into the cluster

## Work flow
- Start master, Then Server, Then Worker
- Every Server and Worker register to Master
- Master decide data partition
- Worker got partition information, get data back, then begin training
- Every one batch finished, push and pull to update parameters
- After finished training, inform master
- Master finished the subsequent work(save parameters etc.)
- Finish training, test result
