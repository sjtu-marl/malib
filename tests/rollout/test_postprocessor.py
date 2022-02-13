import pytest
import random
import numpy as np

from malib.rollout import postprocessor
from malib.utils.episode import EpisodeKey


def gen_episode_with_legal_keys(keys, num_batch=10, num_episode=1, num_agent=1):
    res = []
    for _ in range(num_episode):
        episode = {f"agent_{i}": {} for i in range(num_agent)}
        for a, content in episode.items():
            for k in keys:
                if k in [EpisodeKey.CUR_OBS, EpisodeKey.NEXT_OBS]:
                    content[k] = np.random.rand(num_batch, 3, 2)
                elif k in [EpisodeKey.CUR_STATE, EpisodeKey.NEXT_STATE]:
                    content[k] = np.random.rand(num_batch, 4)
                elif k in [EpisodeKey.REWARD, EpisodeKey.DONE, EpisodeKey.ACTION]:
                    content[k] = np.random.rand(num_batch)
                elif k in [EpisodeKey.ACTION_DIST, EpisodeKey.ACTION_MASK]:
                    content[k] = np.random.rand(num_batch, 4)
                else:
                    raise NotImplementedError(
                        "Initialization for {} not implemented yet.".format(k)
                    )
        res.append(episode)
    return res


def keys_candidate(_type):
    if _type == "default":
        res = [
            EpisodeKey.CUR_OBS,
            EpisodeKey.NEXT_OBS,
            EpisodeKey.ACTION,
            EpisodeKey.ACTION_MASK,
            EpisodeKey.REWARD,
            EpisodeKey.DONE,
        ]
    elif _type == "full":
        res = [
            EpisodeKey.CUR_OBS,
            EpisodeKey.NEXT_OBS,
            EpisodeKey.ACTION,
            EpisodeKey.ACTION_MASK,
            EpisodeKey.REWARD,
            EpisodeKey.DONE,
            EpisodeKey.ACTION_DIST,
            # optional
            EpisodeKey.CUR_STATE,
            EpisodeKey.NEXT_STATE,
        ]
    elif _type == "no_next":
        res = [
            EpisodeKey.CUR_OBS,
            EpisodeKey.ACTION,
            EpisodeKey.ACTION_MASK,
            EpisodeKey.REWARD,
            EpisodeKey.DONE,
            EpisodeKey.ACTION_DIST,
            EpisodeKey.CUR_STATE,
        ]
    return res


@pytest.mark.parametrize(
    "episodes, legal_keys",
    [
        (
            gen_episode_with_legal_keys(keys=keys_candidate("default")),
            keys_candidate("default"),
        ),
        (
            gen_episode_with_legal_keys(
                keys_candidate("full"), num_agent=2, num_episode=2
            ),
            keys_candidate("full"),
        ),
        (
            gen_episode_with_legal_keys(keys_candidate("no_next"), num_agent=2),
            keys_candidate("no_next"),
        ),
    ],
)
@pytest.mark.parametrize("use_gae", [False, True])
class TestHandlers:
    @pytest.fixture(autouse=True)
    def init(self, episodes, legal_keys, use_gae):
        class fake_policy:
            def __init__(self) -> None:
                self.custom_config = {
                    "use_gae": use_gae,
                    "use_critic": False,
                    "gamma": 1.0,
                }

            def value_function(self, *args, **kwargs):
                # fake values
                if len(args) > 0:
                    episode = args[0]
                else:
                    episode = kwargs
                n_batch = len(episode[EpisodeKey.CUR_OBS])
                value = np.zeros((n_batch,))
                return value

        agents = list(episodes[0].keys())
        self.policy_dict = dict.fromkeys(agents, fake_policy())
        self.agents = agents

    def test_compute_acc_reward(self, episodes, legal_keys):
        res = postprocessor.compute_acc_reward(episodes, self.policy_dict)
        # check acc_reward
        for episode in res:
            for agent in self.agents:
                assert agent in episode
            for aid, content in episode.items():
                assert EpisodeKey.ACC_REWARD in content

    def test_compute_advantage(self, episodes, legal_keys):
        last_r = {}
        for agent_episode in episodes:
            for aid, episode in agent_episode.items():
                dones = episode[EpisodeKey.DONE]
                if dones[-1]:
                    last_r[aid] = 0.0
                else:
                    # compute value as last r
                    assert hasattr(self.policy_dict[aid], "value_functon")
                    last_r[aid] = self.policy_dict[aid].value_function(
                        episode, agent_key=aid
                    )
        episodes = postprocessor.compute_value(episodes, self.policy_dict)
        res = postprocessor.compute_advantage(episodes, self.policy_dict, last_r=last_r)

    def test_compute_gae(self, episodes, legal_keys):
        res = postprocessor.compute_gae(episodes, self.policy_dict)

    def test_copy_next_frame(self, episodes, legal_keys):
        # remove next_obs if included
        if EpisodeKey.NEXT_OBS in legal_keys:
            for e in episodes:
                for aid, ae in e.items():
                    ae.pop(EpisodeKey.NEXT_OBS)
        res = postprocessor.copy_next_frame(episodes, None)
        for e in episodes:
            for aid, ae in e.items():
                assert EpisodeKey.NEXT_OBS in ae
                assert ae[EpisodeKey.NEXT_OBS] is not ae[EpisodeKey.CUR_OBS]

    def test_default_processor(self, episodes, legal_keys):
        postprocessor.default_processor(episodes, None)


@pytest.mark.parametrize(
    "_type",
    ["gae", "acc_reward", "advantage", "default", "value", "copy_next_frame", None],
)
def test_getter(_type: str):
    _type = [_type]
    if _type is None:
        with pytest.raises(ValueError):
            for _ in postprocessor.get_postprocessor(_type):
                pass
    else:
        for handler in postprocessor.get_postprocessor(_type):
            if _type[0] == "default":
                assert handler.__name__ == "default_processor"
            elif _type[0] == "copy_next_frame":
                assert handler.__name__ == "copy_next_frame"
            else:
                assert handler.__name__ == f"compute_{_type[0]}"
