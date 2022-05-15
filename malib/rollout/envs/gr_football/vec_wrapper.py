# import os
# import multiprocessing

# import cloudpickle
# import numpy as np

# from abc import ABC, abstractmethod

# from malib.utils.typing import Dict, Optional, List, Union


# class VecEnv(ABC):
#     """
#     An abstract asynchronous, vectorized environment.
#     :param num_envs: (int) the number of environments
#     :param observation_space: (Gym Space) the observation space
#     :param action_space: (Gym Space) the action space
#     """

#     metadata = {"render.modes": ["human", "rgb_array"]}

#     def __init__(
#         self, num_envs, observation_spaces, action_spaces, state_space, possible_agents
#     ):
#         self.num_envs = num_envs
#         self.observation_spaces = observation_spaces
#         self.action_spaces = action_spaces
#         self.state_space = state_space
#         self.possible_agents = possible_agents

#     @abstractmethod
#     def reset(self):
#         """
#         Reset all the environments and return an array of
#         observations, or a tuple of observation arrays.
#         If step_async is still doing work, that work will
#         be cancelled and step_wait() should not be called
#         until step_async() is invoked again.
#         :return: ([int] or [float]) observation
#         """
#         pass

#     @abstractmethod
#     def step_async(self, actions):
#         """
#         Tell all the environments to start taking a step
#         with the given actions.
#         Call step_wait() to get the results of the step.
#         You should not call this if a step_async run is
#         already pending.
#         """
#         pass

#     @abstractmethod
#     def step_wait(self):
#         """
#         Wait for the step taken with step_async().
#         :return: ([int] or [float], [float], [bool], dict) observation, reward, done, information
#         """
#         pass

#     @abstractmethod
#     def close(self):
#         """
#         Clean up the environment's resources.
#         """
#         pass

#     @abstractmethod
#     def seed(self, seed: Optional[int] = None) -> List[Union[None, int]]:
#         """
#         Sets the random seeds for all environments, based on a given seed.
#         Each individual environment will still get its own seed, by incrementing the given seed.
#         :param seed: (Optional[int]) The random seed. May be None for completely random seeding.
#         :return: (List[Union[None, int]]) Returns a list containing the seeds for each individual env.
#             Note that all list elements may be None, if the env does not return anything when being seeded.
#         """
#         pass

#     def step(self, actions):
#         """
#         Step the environments with the given action
#         :param actions: ([int] or [float]) the action
#         :return: ([int] or [float], [float], [bool], dict) observation, reward, done, information
#         """
#         self.step_async(actions)
#         return self.step_wait()


# class CloudpickleWrapper(object):
#     def __init__(self, var):
#         """
#         Uses cloudpickle to serialize contents (otherwise multiprocessing tries to use pickle)
#         :param var: (Any) the variable you wish to wrap for pickling with cloudpickle
#         """
#         self.var = var

#     def __getstate__(self):
#         return cloudpickle.dumps(self.var)

#     def __setstate__(self, obs):
#         self.var = cloudpickle.loads(obs)


# def _worker(remote, parent_remote, env_fn_wrapper):
#     parent_remote.close()
#     env = env_fn_wrapper.var()
#     while True:
#         try:
#             cmd, data = remote.recv()
#             if cmd == "step":
#                 observation, reward, done, info = env.step(data)
#                 if "bool" in done.__class__.__name__ and done:
#                     observation = env.reset()
#                 if all([d.all() for d in done.values()]):
#                     # save final observation where user can get it, then reset
#                     info["terminal_observation"] = observation
#                     observation = env.reset()
#                 remote.send((observation, reward, done, info))
#             elif cmd == "seed":
#                 remote.send(env.seed(data))
#             elif cmd == "reset":
#                 observation = env.reset()
#                 remote.send(observation)
#             elif cmd == "render":
#                 remote.send(env.render(data))
#             elif cmd == "close":
#                 env.close()
#                 remote.close()
#                 break
#             elif cmd == "get_spaces":
#                 remote.send(
#                     (
#                         env.observation_spaces,
#                         env.action_spaces,
#                         env.state_space,
#                         env.possible_agents,
#                     )
#                 )
#             elif cmd == "env_method":
#                 method = getattr(env, data[0])
#                 remote.send(method(*data[1], **data[2]))
#             elif cmd == "get_attr":
#                 remote.send(getattr(env, data))
#             elif cmd == "set_attr":
#                 remote.send(setattr(env, data[0], data[1]))
#             else:
#                 raise NotImplementedError(
#                     "`{}` is not implemented in the worker".format(cmd)
#                 )
#         except EOFError:
#             break


# class SubprocVecEnv(VecEnv):
#     """
#     Creates a multiprocess vectorized wrapper for multiple environments, distributing each environment to its own
#     process, allowing significant speed up when the environment is computationally complex.
#     For performance reasons, if your environment is not IO bound, the number of environments should not exceed the
#     number of logical cores on your CPU.
#     .. warning::
#         Only 'forkserver' and 'spawn' start methods are thread-safe,
#         which is important when TensorFlow sessions or other non thread-safe
#         libraries are used in the parent (see issue #217). However, compared to
#         'fork' they incur a small start-up cost and have restrictions on
#         global variables. With those methods, users must wrap the code in an
#         ``if __name__ == "__main__":`` block.
#         For more information, see the multiprocessing documentation.
#     :param env_fns: ([callable]) A list of functions that will create the environments
#         (each callable returns a `Gym.Env` instance when called).
#     :param start_method: (str) method used to start the subprocesses.
#            Must be one of the methods returned by multiprocessing.get_all_start_methods().
#            Defaults to 'forkserver' on available platforms, and 'spawn' otherwise.
#     """

#     def __init__(self, env_fns, start_method=None):
#         self.waiting = False
#         self.closed = False
#         n_envs = len(env_fns)

#         # In some cases (like on GitHub workflow machine when running tests),
#         # "forkserver" method results in an "connection error" (probably due to mpi)
#         # We allow to bypass the default start method if an environment variable
#         # is specified by the user
#         if start_method is None:
#             start_method = os.environ.get("DEFAULT_START_METHOD")

#         # No DEFAULT_START_METHOD was specified, start_method may still be None
#         if start_method is None:
#             # Fork is not a thread safe method (see issue #217)
#             # but is more user friendly (does not require to wrap the code in
#             # a `if __name__ == "__main__":`)
#             forkserver_available = (
#                 "forkserver" in multiprocessing.get_all_start_methods()
#             )
#             start_method = "forkserver" if forkserver_available else "spawn"
#         ctx = multiprocessing.get_context(start_method)

#         self.remotes, self.work_remotes = zip(
#             *[ctx.Pipe(duplex=True) for _ in range(n_envs)]
#         )
#         self.processes = []
#         for work_remote, remote, env_fn in zip(
#             self.work_remotes, self.remotes, env_fns
#         ):
#             args = (work_remote, remote, CloudpickleWrapper(env_fn))
#             # daemon=True: if the main process crashes, we should not cause things to hang
#             process = ctx.Process(
#                 target=_worker, args=args, daemon=True
#             )  # pytype:disable=attribute-error
#             process.start()
#             self.processes.append(process)
#             work_remote.close()

#         self.remotes[0].send(("get_spaces", None))
#         observation_spaces, action_spaces, state_space, possible_agents = self.remotes[
#             0
#         ].recv()
#         VecEnv.__init__(
#             self,
#             len(env_fns),
#             observation_spaces,
#             action_spaces,
#             state_space,
#             possible_agents,
#         )

#     def step_async(self, actions):
#         for remote, action in zip(self.remotes, _split_dict(actions, self.num_envs)):
#             remote.send(("step", action))
#         self.waiting = True

#     def step_wait(self):
#         results = [remote.recv() for remote in self.remotes]
#         self.waiting = False
#         # obs, rews, dones, infos = zip(*results)
#         # return _flatten_obs(obs, self.observation_space), np.stack(rews), np.stack(dones), infos
#         obs = _merge_list([res[0] for res in results])
#         rews = _merge_list([res[1] for res in results])
#         dones = _merge_list([res[2] for res in results])
#         infos = {k: _merge_list([res[3][k] for res in results]) for k in obs}
#         return obs, rews, dones, infos

#     def seed(self, seed=None):
#         for idx, remote in enumerate(self.remotes):
#             remote.send(("seed", seed + idx if seed else None))
#         obs = [remote.recv() for remote in self.remotes]
#         return _merge_list(obs)

#     def reset(self):
#         for remote in self.remotes:
#             remote.send(("reset", None))
#         obs = [remote.recv() for remote in self.remotes]
#         return _merge_list(obs)

#     def close(self):
#         if self.closed:
#             return
#         if self.waiting:
#             for remote in self.remotes:
#                 remote.recv()
#         for remote in self.remotes:
#             remote.send(("close", None))
#         for process in self.processes:
#             process.join()
#         self.closed = True


# class DummyVecEnv(VecEnv):
#     def __init__(self, env_fns):
#         self.envs = [fn() for fn in env_fns]
#         env_for_space = self.envs[0]
#         VecEnv.__init__(
#             self,
#             len(env_fns),
#             env_for_space.observation_spaces,
#             env_for_space.action_spaces,
#             env_for_space.state_space,
#             env_for_space.possible_agents,
#         )
#         self.actions = None

#     def step_async(self, actions):
#         self.actions = _split_dict(actions, self.num_envs)

#     def step_wait(self):
#         results = [env.step(a) for (a, env) in zip(self.actions, self.envs)]
#         rews = _merge_list([res[1] for res in results])
#         dones = _merge_list([res[2] for res in results])

#         obss = [res[0] for res in results]
#         for i, (env, res) in enumerate(zip(self.envs, results)):
#             done = res[2]
#             if "bool" in done.__class__.__name__ and done:
#                 obss[i] = env.reset()
#             if all([d.all() for d in done.values()]):
#                 # save final observation where user can get it, then reset
#                 obss[i] = env.reset()
#                 res[3]["terminal_observation"] = obss[i]

#         obs = _merge_list(obss)
#         infos = {k: _merge_list([res[3][k] for res in results]) for k in obs}
#         self.actions = None
#         return obs, rews, dones, infos

#     def reset(self):
#         obs = [env.reset() for env in self.envs]
#         return _merge_list(obs)

#     def seed(self, seed=None):
#         obs = [env.seed(seed + i if seed else None) for i, env in enumerate(self.envs)]
#         return _merge_list(obs)

#     def close(self):
#         for env in self.envs:
#             env.close()

#     def render(self, mode="human"):
#         if mode == "rgb_array":
#             return np.array([env.render(mode=mode) for env in self.envs])
#         elif mode == "human":
#             for env in self.envs:
#                 env.render(mode=mode)
#         else:
#             raise NotImplementedError


# def _split_dict(
#     dict_data: Dict[str, np.ndarray], num_envs: int, axis: int = 0
# ) -> List[Dict]:
#     """split a dict of data into a list of dict"""
#     dict_list_data = {}
#     for k, v in dict_data.items():
#         dict_list_data[k] = np.split(v, num_envs, axis)

#     res = [{} for _ in range(num_envs)]
#     for k, v in dict_list_data.items():
#         for i in range(num_envs):
#             res[i][k] = np.squeeze(v[i], axis=axis)
#     return res


# def _merge_list(
#     list_data: List[Dict[str, np.ndarray]],
#     axis: int = 0,
#     expand_dims: bool = True,
# ) -> Dict[str, np.ndarray]:
#     res = {}
#     for k in list_data[0]:
#         if expand_dims:
#             res[k] = np.concatenate(
#                 [np.expand_dims(ld[k], axis=axis) for ld in list_data], axis=axis
#             )
#         else:
#             res[k] = np.concatenate([ld[k] for ld in list_data], axis=axis)

#     return res


# if __name__ == "__main__":
#     # action = {"team_0": np.ones((2, 4, 1)), "team_1": np.zeros((2,4,1))}
#     # action_split = _split_dict(action, 2, 0)
#     # print(action_split)
#     # print("="*40)
#     # print(_merge_list(action_split))
#     from malib.rollout.envs.gr_football import default_config
#     from malib.rollout.envs.gr_football import BaseGFootBall, ParameterSharingWrapper

#     def env_fn():
#         env = BaseGFootBall(**default_config)
#         env = ParameterSharingWrapper(env, lambda x: x[:6])
#         return env

#     vec_env = SubprocVecEnv([env_fn for _ in range(2)])
#     obs = vec_env.reset()
#     print(obs["team_0"].shape)
#     action = {k: np.zeros((2, 4)).astype(np.int) for k in obs}
#     vec_env.step(action)
