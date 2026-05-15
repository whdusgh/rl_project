from env.minigrid_env import make_env

from agents.ppo_agent import create_ppo


env = make_env()

print("\nEnvironment created.")

obs, _ = env.reset()

print("\nObservation keys:")
print(obs.keys())

print("\nImage shape:")
print(obs["image"].shape)

print("\nReasoning shape:")
print(obs["reasoning"].shape)


model = create_ppo(env)

print("\nPPO model created.")

model.learn(total_timesteps=5000)

print("\nTraining finished.")

model.save("ppo_llm")