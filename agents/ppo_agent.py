from stable_baselines3 import PPO

from models.multimodal_extractor import MultiModalExtractor


def create_ppo(env):

    policy_kwargs = dict(

        features_extractor_class=MultiModalExtractor,

        features_extractor_kwargs=dict(
            features_dim=256
        )
    )

    model = PPO(
        "MultiInputPolicy",
        env,
        policy_kwargs=policy_kwargs,
        verbose=1,
        learning_rate=3e-4,
        n_steps=1024,
        batch_size=64,
        gamma=0.99,
        tensorboard_log="./logs/"
    )

    return model