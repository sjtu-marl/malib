from malib.envs.vector_env import VectorEnv
from malib.envs import GymEnv

env = GymEnv(
    env_id="Pendulum-v0"
)
vec_env = VectorEnv.from_envs([env], {})
print(1)
