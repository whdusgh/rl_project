import gymnasium as gym
import minigrid

from minigrid.wrappers import ImgObsWrapper
from env.llm_wrapper import LLMObservationWrapper

def make_env():
    env = gym.make(
        "MiniGrid-DoorKey-6x6-v0",
        render_mode="rgb_array"
    )
    env = ImgObsWrapper(env)
    env = LLMObservationWrapper(env)
    return env