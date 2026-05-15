from env.minigrid_env import make_env

env = make_env()

obs, _ = env.reset()

print("\n===== Observation Keys =====")
print(obs.keys())

print("\n===== Image Shape =====")
print(obs["image"].shape)

print("\n===== Reasoning Shape =====")
print(obs["reasoning"].shape)

print("\n===== Reasoning Sample =====")
print(obs["reasoning"][:10])