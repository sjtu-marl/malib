import pytest

from pytest_mock import MockerFixture

from malib.backend.coordinator.task import CoordinatorServer
from malib.utils.typing import (
    AgentInvolveInfo,
    EvaluateResult,
    TaskRequest,
    TaskType,
    TrainingFeedback,
)
from malib.agent.agent_interface import AgentFeedback

from .mixin import ServerTestMixin


@pytest.mark.parametrize(
    "yaml_path,",
    ["examples/configs/mpe/ddpg_simple_spread.yaml"],
)
class TestRequestHandle(ServerTestMixin):
    def init_custom(self, exp_cfg, config):
        possible_agents = config["env_description"]["possible_agents"]
        self.server = CoordinatorServer(exp_cfg=exp_cfg, **config)
        self.request_template = TaskRequest(
            task_type=TaskType.OPTIMIZE,
            content=AgentFeedback(
                id="test",
                trainable_pairs={
                    agent: ("policy_0", None) for agent in possible_agents
                },
                statistics={},
            ),
            state_id="test",
        )
        self.server.start()

    def test_handler_execution(self, mocker: MockerFixture):
        # ============ training manager mocker =================
        training_feedback = TrainingFeedback(
            agent_involve_info=AgentInvolveInfo(
                training_handler="test",
                populations={
                    k: [v]
                    for k, v in self.request_template.content.trainable_pairs.items()
                },
                trainable_pairs=self.request_template.content.trainable_pairs,
            ),
            statistics={},
        )
        mocked_training_manager = mocker.patch(
            "malib.manager.training_manager.TrainingManager"
        )
        self.server.training_manager.init = mocker.patch.object(
            mocked_training_manager, "init", return_value=None
        )
        self.server.training_manager.retrieve_information = mocker.patch.object(
            mocked_training_manager,
            "retrieve_information",
            return_value=TaskRequest(
                task_type=None,
                content=training_feedback,
                state_id=self.request_template.state_id,
            ),
        )
        self.server.training_manager.optimize = mocker.patch.object(
            mocked_training_manager, "optimize"
        )
        self.request_template.task_type = TaskType.OPTIMIZE
        self.server.request(self.request_template)
        self.server.training_manager.retrieve_information.assert_called_once()
        self.server.training_manager.optimize.assert_called_once()

        # ========= simulation =========
        self.server.payoff_manager.get_pending_matchups = mocker.patch.object(
            self.server.payoff_manager, "get_pending_matchups"
        )
        self.server.gen_simulation_task = mocker.patch.object(
            self.server, "gen_simulation_task"
        )
        self.request_template.task_type = TaskType.SIMULATION
        self.server.request(self.request_template)

        # ========== evaluation ==========
        self.request_template.task_type = TaskType.EVALUATE
        self.request_template.content = training_feedback
        self.server.request(self.request_template)

        # ======== update payoff table
        self.request_template.task_type = TaskType.UPDATE_PAYOFFTABLE
        self.server.task_mode = "gt"
        self.server.payoff_manager.update_payoff = mocker.patch.object(
            self.server.payoff_manager, "update_payoff"
        )
        self.server.payoff_manager.check_done = mocker.patch.object(
            self.server.payoff_manager, "check_done", return_value=True
        )
        self.server.payoff_manager.compute_equilibrium = mocker.patch.object(
            self.server.payoff_manager, "compute_equilibrium"
        )
        self.server.payoff_manager.update_equilibrium = mocker.patch.object(
            self.server.payoff_manager, "update_equilibrium"
        )

        # true or not
        self.server._hyper_evaluator.evaluate = mocker.patch.object(
            self.server._hyper_evaluator,
            "evaluate",
            return_valu={EvaluateResult.CONVERGED: True},
        )
        self.server.request(self.request_template)

        # self.server._hyper_evaluator.evaluate.reset_mock(return_value={EvaluateResult.CONVERGED: False})

        # ============= rollout ===================
        self.request_template.task_type = TaskType.ROLLOUT
        self.server.payoff_manager.get_equilibrium = mocker.patch.object(
            self.server.payoff_manager, "get_equilibrium"
        )
        self.server.rollout_manager.rollout = mocker.patch.object(
            self.server.rollout_manager, "rollout"
        )
        self.server.request(self.request_template)
