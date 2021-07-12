"""
Implementation of centralized agent learning, one team (many policies, many agents) one trainer.
"""

from copy import deepcopy

import gym

from malib.algorithm import get_algorithm_space
from malib.utils import metrics
from malib.utils.typing import (
    Dict,
    Any,
    AgentID,
    PolicyID,
    MetricEntry,
    Callable,
    Tuple,
)
from malib.agent.ctde_agent import CTDEAgent
from malib.algorithm.common.policy import Policy


class CentralizedAgent(CTDEAgent):
    def save(self, model_dir):
        pass

    def load(self, model_dir):
        pass

    def __init__(
        self,
        assign_id: str,
        env_desc: Dict[str, Any],
        algorithm_candidates: Dict[str, Any],
        training_agent_mapping: Callable,
        observation_spaces: Dict[AgentID, gym.spaces.Space],
        action_spaces: Dict[AgentID, gym.spaces.Space],
        exp_cfg: Dict[str, Any],
        population_size: int = -1,
        algorithm_mapping: Callable = None,
    ):
        assert hasattr(env_desc, "teams"), "Env description should specify the teams"

        CTDEAgent.__init__(
            self,
            assign_id=assign_id,
            env_desc=env_desc,
            algorithm_candidates=algorithm_candidates,
            training_agent_mapping=training_agent_mapping,
            observation_spaces=observation_spaces,
            action_spaces=action_spaces,
            exp_cfg=exp_cfg,
            population_size=population_size,
            algorithm_mapping=algorithm_mapping,
        )

        self._teams = deepcopy(env_desc["teams"])

        self._agent_to_team = {}
        _ = [
            self._agent_to_team.update({v: k for v in ids})
            for k, ids in self._teams.items()
        ]

    def optimize(
        self,
        policy_ids: Dict[AgentID, PolicyID],
        batch: Dict[AgentID, Any],
        training_config: Dict[str, Any],
    ) -> Dict[AgentID, Dict[str, MetricEntry]]:
        res = {}
        # extract a group of policies
        t_policies = {}
        for env_agent_id, pid in policy_ids.items():
            t_policies[env_agent_id] = self.policies[pid]
        # get trainer by team
        for tid, env_agent_ids in self._teams.items():
            trainer = self.get_trainer(tid)
            trainer.reset(t_policies, training_config)
            # filter batch with env_agent_ids
            _batch = {aid: data for aid, data in batch.items()}
            _batch = trainer.preprocess(_batch, other_policies=t_policies)
            res.update(metrics.to_metric_entry(trainer.optimize(_batch), prefix=tid))
        return res

    def add_policy_for_agent(
        self, env_agent_id: AgentID, trainable: bool
    ) -> Tuple[PolicyID, Policy]:
        assert env_agent_id in self.agent_group(), (env_agent_id, self.agent_group())
        algorithm_conf = self.get_algorithm_config(env_agent_id)
        pid = self.default_policy_id_gen(algorithm_conf)

        if pid in self.policies:
            return pid, self.policies[pid]
        else:
            algorithm_space = get_algorithm_space(algorithm_conf["name"])
            custom_config = algorithm_conf.get("custom_config", {})
            # group spaces into a new space
            team_agents = self._teams[self._agent_to_team[env_agent_id]]
            custom_config.update(
                {
                    "global_state_space": gym.spaces.Dict(
                        {
                            "obs": gym.spaces.Dict(
                                **{
                                    aid: self._observation_spaces[aid]
                                    for aid in team_agents
                                }
                            ),
                            "act": gym.spaces.Discrete(len(team_agents)),
                        }
                    )
                }
            )

            policy = algorithm_space.policy(
                registered_name=algorithm_conf["name"],
                observation_space=self._observation_spaces[env_agent_id],
                action_space=self._action_spaces[env_agent_id],
                model_config=algorithm_conf.get("model_config", {}),
                custom_config=custom_config,
            )

            # mapping agent to tid
            for tid, env_agent_ids in self._teams.items():
                if env_agent_id in env_agent_ids and tid not in self._trainers:
                    self._trainers[tid] = algorithm_space.trainer(tid)
                    # assign eam agents to it
                    self._trainers[tid].agents = env_agent_ids.copy()

            self.register_policy(pid, policy)
            return pid, policy
