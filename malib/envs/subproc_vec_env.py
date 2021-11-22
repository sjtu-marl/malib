"""A VecEnv that uses subprocesses for each underlying environment"""

from malib.backend.datapool.offline_dataset_server import Episode
from malib.utils.typing import Dict, AgentID, Any, List, Tuple

from typing import Dict, Sequence, Optional, List, Union
import multiprocessing
import cloudpickle
import os
import numpy as np


def _merge_list(
    list_data: List[Dict[str, np.ndarray]],
    axis: int = 0,
    expand_dims: bool = True,
) -> Dict[str, np.ndarray]:
    res = {}
    for k in list_data[0]:
        if isinstance(list_data[0][k], dict) and k != "info":
            res[k] = _merge_list([ld[k] for ld in list_data])
        elif k == "info":
            return [ld[k] for ld in list_data]
        else:
            if expand_dims:
                res[k] = np.concatenate(
                    [np.expand_dims(ld[k], axis=axis) for ld in list_data], axis=axis
                )
            else:
                res[k] = np.concatenate([ld[k] for ld in list_data], axis=axis)
    return res


def _split_dict(
    dict_data: Dict[str, np.ndarray], num_envs: int, axis: int = 0
) -> List[Dict]:
    """split a dict of data into a list of dict"""
    dict_list_data = {}
    for k, v in dict_data.items():
        dict_list_data[k] = np.split(v, num_envs, axis)

    res = [{} for _ in range(num_envs)]
    for k, v in dict_list_data.items():
        for i in range(num_envs):
            res[i][k] = np.squeeze(v[i], axis=axis)
    return res


class CloudpickleWrapper(object):
    def __init__(self, var):
        """
        Uses cloudpickle to serialize contents (otherwise multiprocessing tries to use pickle)
        :param var: (Any) the variable you wish to wrap for pickling with cloudpickle
        """
        self.var = var

    def __getstate__(self):
        return cloudpickle.dumps(self.var)

    def __setstate__(self, obs):
        self.var = cloudpickle.loads(obs)


def _worker(remote, parent_remote, env_fn_wrapper):
    parent_remote.close()
    env = env_fn_wrapper.var()
    while True:
        try:
            cmd, data = remote.recv()
            if cmd == "step":
                rets = env.step(data)
                done, info = rets[Episode.DONE], rets[Episode.INFO]
                if all([d.all() for d in done.values()]):
                    # save final observation where user can get it, then reset
                    info["episode_info"] = env.episode_info
                    new_rets = env.reset()
                    rets.update(new_rets)

                remote.send(rets)
            elif cmd == "seed":
                remote.send(env.seed(data))
            elif cmd == "reset":
                remote.send(env.reset())
            elif cmd == "render":
                remote.send(env.render(data))
            elif cmd == "close":
                env.close()
                remote.close()
                break
            elif cmd == "get_spaces":
                remote.send(
                    (
                        env.observation_spaces,
                        env.action_spaces,
                        env.state_space,
                        env.possible_agents,
                    )
                )
            # FIXME(ziyu): not correct under new classmethod
            # elif cmd == "env_method":
            #     method = getattr(env, data[0])
            #     remote.send(method(*data[1], **data[2]))
            elif cmd == "get_attr":
                remote.send(getattr(env, data))
            elif cmd == "set_attr":
                remote.send(setattr(env, data[0], data[1]))
            else:
                raise NotImplementedError(
                    "`{}` is not implemented in the worker".format(cmd)
                )
        except EOFError:
            break


class SubprocVecEnv:
    def __init__(
        self,
        observation_spaces,
        action_spaces,
        creator,
        configs,
        num_envs=0,
        fragment_length=1000,
        start_method=None,
    ):
        self.waiting = False
        self.closed = False
        self.env_fn = lambda: creator(**configs)

        # In some cases (like on GitHub workflow machine when running tests),
        # "forkserver" method results in an "connection error" (probably due to mpi)
        # We allow to bypass the default start method if an environment variable
        # is specified by the user
        if start_method is None:
            start_method = os.environ.get("DEFAULT_START_METHOD")

        # No DEFAULT_START_METHOD was specified, start_method may still be None
        if start_method is None:
            # Fork is not a thread safe method (see issue #217)
            # but is more user friendly (does not require to wrap the code in
            # a `if __name__ == "__main__":`)
            forkserver_available = (
                "forkserver" in multiprocessing.get_all_start_methods()
            )
            start_method = "forkserver" if forkserver_available else "spawn"
        self.ctx = multiprocessing.get_context(start_method)
        self.remotes, self.work_remotes = [], []
        self.processes = []
        

        # self.remotes[0].send(("get_spaces", None))
        # observation_spaces, action_spaces, state_space, possible_agents = self.remotes[
        #     0
        # ].recv()

        env_for_spec = self.env_fn()
        self.observation_spaces = env_for_spec.observation_spaces
        self.action_spaces = env_for_spec.action_spaces
        self.state_space = env_for_spec.state_space
        self.possible_agents = env_for_spec.possible_agents
        self.trainable_agents = env_for_spec.possible_agents # FIXME(ziyu)
        self._fragment_length = fragment_length


        self.is_sequential = False 
        # ziyu: this should be False, vec env for sequential is not available now.
        self.num_envs = 0
        self.add_env(num_envs)
    
    def add_env(self, num_env):
        new_remotes, new_work_remotes = zip(
            *[self.ctx.Pipe(duplex=True) for _ in range(num_env)]
        )
        for work_remote, remote in zip(
            new_remotes, new_work_remotes
        ):
            args = (work_remote, remote, CloudpickleWrapper(self.env_fn))
            # daemon=True: if the main process crashes, we should not cause things to hang
            process = self.ctx.Process(
                target=_worker, args=args, daemon=True
            )  # pytype:disable=attribute-error
            process.start()
            self.processes.append(process)
            work_remote.close()
            self.remotes.append(remote)
            self.work_remotes.append(work_remote)
        self.num_envs += num_env

    def step(self, actions):
        self._step_cnt += 1
        self.step_async(actions)
        return self.step_wait()

    def step_async(self, actions):
        for remote, action in zip(self.remotes, _split_dict(actions, self.num_envs)):
            remote.send(("step", action))
        self.waiting = True

    def step_wait(self):
        results = [remote.recv() for remote in self.remotes]
        self.waiting = False
        feed_back_needed = []
        for res in results:
            if any(d.any() for d in res[Episode.DONE].values()):
                self.episode_infos.append(res[Episode.INFO]["episode_info"])
        return _merge_list(results)

    def seed(self, seed=None):
        self._step_cnt = 0
        for idx, remote in enumerate(self.remotes):
            remote.send(("seed", seed + idx if seed else None))
        obs = [remote.recv() for remote in self.remotes]
        return {Episode.CUR_OBS: _merge_list(obs)}

    def reset(
        self,
        limits: int = None,  # XXX(ziyu): unnecessary
        fragment_length: int = None,
        env_reset_kwargs: Dict = None,
    ):
        self.episode_infos = []
        assert limits == self.num_envs, (limits, self.num_envs)
        self._step_cnt = 0
        self._fragment_length = fragment_length or self._fragment_length

        for remote in self.remotes:
            remote.send(("reset", None))
        rets_list = [remote.recv() for remote in self.remotes]
        return _merge_list(rets_list)

    def close(self):
        if self.closed:
            return
        if self.waiting:
            for remote in self.remotes:
                remote.recv()
        for remote in self.remotes:
            remote.send(("close", None))
        for process in self.processes:
            process.join()
        self.closed = True

    def render(self):
        raise NotImplementedError

    def is_terminated(self):
        return self._step_cnt >= self._fragment_length

    def add_envs(self, envs=None, num=0):
        # TODO(ziyu): find ways to add/remove env
        raise NotImplementedError

    @classmethod
    def from_envs(cls, envs: List, config: Dict[str, Any]):
        """Generate vectorization environment from exisiting environments."""
        raise NotImplementedError

    @property
    def batched_step_cnt(self) -> int:
        return self._step_cnt
