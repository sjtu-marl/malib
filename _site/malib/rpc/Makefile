objects = control data log exprmanager mongo

all : $(objects)


control : proto/control_pb2_grpc.py proto/control_pb2.py
proto/control_pb2_grpc.py proto/control_pb2.py : 
	python3 -m grpc_tools.protoc --python_out=. --grpc_python_out=. -I. proto/control.proto

data : proto/data_pb2_grpc.py proto/data_pb2.py
proto/data_pb2_grpc.py proto/data_pb2.py : 
	python3 -m grpc_tools.protoc --python_out=. --grpc_python_out=. -I. proto/data.proto

log : proto/log_pb2_grpc.py proto/log_pb2.py
proto/log_pb2_grpc.py proto/log_pb2.py : 
	python3 -m grpc_tools.protoc --python_out=. --grpc_python_out=. -I. proto/log.proto

exprmanager : proto/exprmanager_pb2_grpc.py proto/exprmanager_pb2.py
proto/exprmanager_pb2_grpc.py proto/exprmanager_pb2.py :
	python3 -m grpc_tools.protoc --python_out=. --grpc_python_out=. -I. proto/exprmanager.proto

mongo : $(addprefix proto/mongo_, pb2_grpc.py pb2.py)
proto/mongo_pb2_grpc.py proto/mongo_pb2.py :
	python3 -m grpc_tools.protoc --python_out=. --grpc_python_out=. -I. proto/mongo.proto

.PHONY: clean all control data log exprmanager mongo

clean : 
	for obj in $(objects); do rm -rf proto/$$obj*.py; done