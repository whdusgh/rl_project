from env.minigrid_env import make_env

from llm.state_parser import parse_observation


env = make_env()

obs, _ = env.reset()

image = obs["image"]

state_text = parse_observation(image)

print("\nState Description:")
print(state_text)