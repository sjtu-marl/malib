from typing import Any, Dict

import traceback

from malib.utils.logging import Logger
from malib.common.manager import Manager


class League:
    def __init__(self, training_manager: Manager, rollout_manager: Manager) -> None:
        self.training_manager = training_manager
        self.rollout_manager = rollout_manager

    def get_results(self) -> Dict[str, Dict[str, Any]]:
        """Retrieve results from rollout and training manager.

        Returns:
            Dict[str, Dict[str, Any]]: A dict of results, which contains rollout and training results.
        """

        rollout_results = []
        training_results = []

        try:
            while True:
                for result in self.rollout_manager.get_results():
                    rollout_results.append(result)
                for result in self.training_manager.get_results():
                    training_results.append(result)
        except KeyboardInterrupt:
            Logger.info("Keyboard interruption was detected, recalling resources ...")
        except RuntimeError:
            Logger.error(traceback.format_exc())
        except Exception:
            Logger.error(traceback.format_exc())

        return {"rollout": rollout_results, "training": training_results}
    
    def terminate(self):
        self.training_manager.terminate()
        self.rollout_manager.terminate()
