import traceback

from malib.utils.logging import Logger
from malib.common.manager import Manager


class League:
    def __init__(self, training_manager: Manager, rollout_manager: Manager) -> None:
        self.training_manager = training_manager
        self.rollout_manager = rollout_manager

    def get_results(self):
        try:
            while True:
                # TODO(ming): check whether done
                raise NotImplementedError
        except KeyboardInterrupt:
            Logger.info("Keyboard interruption was detected, recalling resources ...")
        except RuntimeError:
            Logger.error(traceback.format_exc())
        except Exception:
            Logger.error(traceback.format_exc())
