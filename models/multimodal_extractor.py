import torch
import torch.nn as nn

from stable_baselines3.common.torch_layers import BaseFeaturesExtractor


class MultiModalExtractor(BaseFeaturesExtractor):

    def __init__(self, observation_space, features_dim=256):

        super().__init__(observation_space, features_dim)

        # -----------------------------
        # Image observation shape
        # -----------------------------
        image_shape = observation_space["image"].shape

        n_input_channels = image_shape[0]

        # -----------------------------
        # CNN for image feature extraction
        # -----------------------------
        self.cnn = nn.Sequential(

            nn.Conv2d(
                n_input_channels,
                32,
                kernel_size=3,
                padding=1
            ),
            nn.ReLU(),

            nn.Conv2d(
                32,
                64,
                kernel_size=3,
                padding=1
            ),
            nn.ReLU(),

            nn.Flatten()
        )

        # -----------------------------
        # CNN output dimension 계산
        # -----------------------------
        with torch.no_grad():

            sample = torch.zeros((1, *image_shape))

            n_flatten = self.cnn(sample).shape[1]

        # -----------------------------
        # Image feature projector
        # -----------------------------
        self.image_linear = nn.Sequential(

            nn.Linear(n_flatten, 128),
            nn.ReLU()
        )

        # -----------------------------
        # Qwen reasoning latent projector
        # -----------------------------
        self.reasoning_linear = nn.Sequential(

            nn.Linear(896, 128),
            nn.ReLU()
        )

        # -----------------------------
        # Gate Network
        # Input:
        #   image_feature + reasoning_feature
        # Output:
        #   trust score (0~1)
        # -----------------------------
        self.gate = nn.Sequential(

            nn.Linear(256, 64),
            nn.ReLU(),

            nn.Linear(64, 1),

            nn.Sigmoid()
        )

    def forward(self, observations):

        # -----------------------------
        # Input observations
        # -----------------------------
        image = observations["image"]

        reasoning = observations["reasoning"]

        # -----------------------------
        # Image feature extraction
        # -----------------------------
        image_feature = self.image_linear(
            self.cnn(image)
        )

        # -----------------------------
        # Reasoning latent projection
        # -----------------------------
        reasoning_feature = self.reasoning_linear(
            reasoning
        )

        # -----------------------------
        # Gate input
        # state(image) + reasoning
        # -----------------------------
        gate_input = torch.cat(
            [image_feature, reasoning_feature],
            dim=1
        )

        # -----------------------------
        # Gate value
        # -----------------------------
        gate_value = self.gate(gate_input)

        # -----------------------------
        # Apply gate to reasoning
        # -----------------------------
        gated_reasoning = reasoning_feature * gate_value

        # -----------------------------
        # Final multimodal feature
        # -----------------------------
        combined = torch.cat(
            [image_feature, gated_reasoning],
            dim=1
        )

        # -----------------------------
        # Debug print
        # -----------------------------
        print("\nGate value:")
        print(gate_value.mean().item())

        return combined