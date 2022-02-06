import ray

from malib import settings


class ServerMixin:
    def init_coordinator(self):
        server = None

        try:
            server = ray.get_actor(settings.COORDINATOR_SERVER_ACTOR)
        except ValueError:
            from tests.coordinator import FakeCoordinator

            server = FakeCoordinator.options(
                name=settings.COORDINATOR_SERVER_ACTOR
            ).remote()

        return server

    def init_dataserver(self):
        server = None

        try:
            server = ray.get_actor(settings.OFFLINE_DATASET_ACTOR)
        except ValueError:
            from tests.dataset import FakeDataServer

            server = FakeDataServer.options(
                name=settings.OFFLINE_DATASET_ACTOR
            ).remote()

        return server

    def init_parameter_server(self):
        server = None
        try:
            server = ray.get_actor(settings.PARAMETER_SERVER_ACTOR)
        except ValueError:
            from tests.parameter_server import FakeParameterServer

            server = FakeParameterServer.options(
                name=settings.PARAMETER_SERVER_ACTOR
            ).remote()
        return server
