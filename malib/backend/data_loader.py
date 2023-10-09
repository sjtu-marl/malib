from typing import Dict, Any, Union

import pyarrow.flight as flight

from malib.utils.logging import Logger
from malib.utils.general import find_free_port


class FlightServer(flight.FlightServerBase):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

    def list_flights(
        self, context: flight.ServerCallContext, criteria: bytes
    ) -> flight.FlightInfo:
        info = flight.FlightInfo(...)
        yield info

    def do_action(self, context: flight.ServerCallContext, action: flight.Action):
        """Execute a custom action. This method should return an iterator, or it should be a generator. Applications should override this method to implement their own behavior. The default method raises a NotImplementedError.

        Args:
            context (_type_): _description_
            action (_type_): _description_

        Raises:
            NotImplementedError: _description_
        """

        raise NotImplementedError

    def do_exchange(
        self,
        context: flight.ServerCallContext,
        descriptor: flight.FlightDescriptor,
        reader: flight.MetadataRecordBatchReader,
        writer: flight.MetadataRecordBatchWriter,
    ):
        raise NotImplementedError

    def do_put(
        self,
        context: flight.ServerCallContext,
        descriptor: flight.FlightDescriptor,
        reader: flight.MetadataRecordBatchReader,
        writer: flight.FlightMetadataWriter,
    ):
        """Write data to a flight."""


class DataCommunicator:
    def __init__(self, flight_server_address: str) -> None:
        self.flight_conn: flight.FlightClient = flight.connect(flight_server_address)

    def send(self, data):
        self.flight_conn.do_put(data)

    def get(self, batch_size: int):
        raise NotImplementedError

    def close(self):
        self.flight_conn.close()


if __name__ == "__main__":
    port = find_free_port()
    flight_server = FlightServer(f"grpc://0.0.0.0:{port}")
    Logger.info(f"Flight server listening on {port}")
    flight_server.serve()
