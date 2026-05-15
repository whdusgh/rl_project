import torch

from env.minigrid_env import make_env
from models.multimodal_extractor import MultiModalExtractor

env = make_env()

obs, _ = env.reset()

extractor = MultiModalExtractor(
    env.observation_space
)

obs_tensor = {

    "image": torch.tensor(obs["image"]).float().unsqueeze(0),

    "reasoning": torch.tensor(obs["reasoning"]).float().unsqueeze(0)
}

features = extractor(obs_tensor)

print("\n===== Feature Shape =====")
print(features.shape)