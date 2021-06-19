import grpc
from concurrent import futures

import node_server as s
import node_client as c

if __name__ == "__main__":
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
    servicer = s.NodeServicer
