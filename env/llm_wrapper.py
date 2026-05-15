import gymnasium as gym
import numpy as np

from gymnasium import spaces

from llm.state_parser import parse_observation

from llm.llm_generator import LLMGenerator

from llm.qwen_encoder import QwenEncoder


class LLMObservationWrapper(gym.ObservationWrapper):

    def __init__(self, env):

        super().__init__(env)

        # -----------------------------
        # LLM generator
        # -----------------------------
        self.generator = LLMGenerator()

        # -----------------------------
        # Qwen encoder
        # -----------------------------
        self.encoder = QwenEncoder()

        # -----------------------------
        # Observation space
        # -----------------------------
        self.observation_space = spaces.Dict({

            "image": env.observation_space,

            "reasoning": spaces.Box(
                low=-np.inf,
                high=np.inf,
                shape=(896,),
                dtype=np.float32
            )
        })

    def observation(self, obs):

        # -----------------------------
        # State → text
        # -----------------------------
        state_text = parse_observation(obs)

        print("\nState Text:")
        print(state_text)

        # -----------------------------
        # LLM reasoning generation
        # -----------------------------
        reasoning_text = self.generator.generate_reasoning(
            state_text
        )

        print("\nGenerated Reasoning:")
        print(reasoning_text)

        # -----------------------------
        # Reasoning → latent
        # -----------------------------
        reasoning_embedding = self.encoder.encode(
            reasoning_text
        )

        reasoning_embedding = reasoning_embedding.numpy().astype(
            np.float32
        )

        return {

            "image": obs,

            "reasoning": reasoning_embedding
        }