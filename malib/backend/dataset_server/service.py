import threading
import traceback
import pickle

from . import data_pb2_grpc
from . import data_pb2
from . import feature


class DatasetServer(data_pb2_grpc.SendDataServicer):
    def __init__(
        self,
        feature_handler: feature.BaseFeature,
        service_event: threading.Event = None,
    ) -> None:
        super().__init__()
        self.feature_handler = feature_handler
        self.service_event = service_event

    def Collect(self, request, context):
        try:
            data = pickle.loads(request.data)
            self.feature_handler.safe_put(data)
            message = "success"
        except Exception as e:
            message = traceback.format_exc()

        return data_pb2.Reply(message=message)
