# _TimeStep = collections.namedtuple(
#     "_TimeStep", "observation, action_mask, reward, action, done, action_dist"
# )

# def sequential(
#     env: AECEnv,
#     num_episodes: int,
#     agent_interfaces: Dict[AgentID, AgentInterface],
#     fragment_length: int,
#     max_step: int,
#     behavior_policies: Dict[AgentID, PolicyID],
#     buffer_desc: BufferDescription,
#     send_interval: int = 50,
#     dataset_server: ray.ObjectRef = None,
# ):
#     """Rollout in sequential manner"""

#     cnt = 0

#     if buffer_desc is not None:
#         agent_filters = buffer_desc.agent_id
#         agent_buffers = {agent: None for agent in agent_filters}
#     else:
#         agent_filters = list(agent_interfaces.keys())
#         agent_buffers = None

#     total_cnt = {agent: 0 for agent in agent_filters}
#     mean_episode_reward = collections.defaultdict(list)
#     mean_episode_len = collections.defaultdict(list)
#     win_rate = collections.defaultdict(list)

#     while any(
#         [agent_total_cnt < fragment_length for agent_total_cnt in total_cnt.values()]
#     ):
#         env.reset()
#         cnt = collections.defaultdict(lambda: 0)
#         tmp_buffer = collections.defaultdict(list)
#         episode_reward = collections.defaultdict(lambda: 0.0)

#         for aid in env.agent_iter(max_iter=max_step):
#             observation, reward, done, info = env.last()
#             action_mask = np.asarray(observation["action_mask"])

#             # observation has been transferred
#             observation = agent_interfaces[aid].transform_observation(
#                 observation, behavior_policies[aid]
#             )
#             if not done:
#                 action, action_dist, _ = agent_interfaces[aid].compute_action(
#                     observation,
#                     action_mask=action_mask,
#                     policy_id=behavior_policies[aid],
#                 )
#                 # convert action to scalar
#             else:
#                 info["policy_id"] = behavior_policies[aid]
#                 action = None
#             env.step(action)

#             if dataset_server and aid in agent_filters:
#                 tmp_buffer[aid].append(
#                     _TimeStep(
#                         observation,
#                         action_mask,
#                         reward,
#                         action
#                         if action is not None
#                         else env.action_spaces[aid].sample(),
#                         done,
#                         action_dist,
#                     )
#                 )
#             episode_reward[aid] += reward
#             cnt[aid] += 1

#             if all([agent_cnt >= fragment_length for agent_cnt in cnt.values()]):
#                 break
#         winner, max_reward = None, -float("inf")
#         total_cnt = {aid: v + cnt[aid] for aid, v in total_cnt.items()}

#         for k, v in episode_reward.items():
#             mean_episode_reward[k].append(v)
#             mean_episode_len[k].append(cnt[k])
#             if v > max_reward:
#                 winner = k
#                 max_reward = v
#         for k in agent_filters:
#             if k == winner:
#                 win_rate[winner].append(1)
#             else:
#                 win_rate[k].append(0)

#     Logger.debug("agent total_cnt: %s fragment length: %s", total_cnt, fragment_length)
#     if dataset_server:
#         shuffle_idx = np.random.permutation(fragment_length)
#         for player, data_tups in tmp_buffer.items():
#             (
#                 observations,
#                 action_masks,
#                 pre_rewards,
#                 actions,
#                 dones,
#                 action_dists,
#             ) = tuple(map(np.stack, list(zip(*data_tups))))

#             rewards = pre_rewards[1:].copy()
#             dones = dones[1:].copy()
#             next_observations = observations[1:].copy()
#             next_action_masks = action_masks[1:].copy()

#             observations = observations[:-1].copy()
#             action_masks = action_masks[:-1].copy()
#             actions = actions[:-1].copy()
#             action_dists = action_dists[:-1].copy()

#             agent_buffers[player] = {
#                 Episode.CUR_OBS: observations[shuffle_idx],
#                 Episode.NEXT_OBS: next_observations[shuffle_idx],
#                 Episode.REWARD: rewards[shuffle_idx],
#                 Episode.ACTION: actions[shuffle_idx],
#                 Episode.DONE: dones[shuffle_idx],
#                 Episode.ACTION_DIST: action_dists[shuffle_idx],
#                 Episode.ACTION_MASK: action_masks[shuffle_idx],
#                 Episode.NEXT_ACTION_MASK: next_action_masks[shuffle_idx],
#             }
#         buffer_desc.batch_size = fragment_length
#         buffer_desc.data = None
#         while indices is None:
#             batch = ray.get(dataset_server.get_producer_index(buffer_desc))
#             indices = batch.data
#         buffer_desc.data = agent_buffers
#         buffer_desc.indices = indices
#         dataset_server.save.remote(buffer_desc)

#     results = {
#         f"total_reward/{k}": v
#         for k, v in mean_episode_reward.items()
#         if k in agent_filters
#     }
#     results.update(
#         {f"step_cnt/{k}": v for k, v in mean_episode_len.items() if k in agent_filters}
#     )
#     results.update(
#         {f"win_rate/{k}": v for k, v in win_rate.items() if k in agent_filters}
#     )

#     # aggregated evaluated results groupped in agent wise
#     return results, sum(total_cnt.values())
