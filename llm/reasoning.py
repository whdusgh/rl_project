import random


def generate_reasoning(obs):

    """
    일부러 good / bad reasoning 섞기
    """

    good_reasonings = [

        "Find the key first.",

        "Open the door carefully.",

        "Go to the goal.",

        "Explore the environment."
    ]

    bad_reasonings = [

        "Ignore the key and walk randomly.",

        "Avoid the door completely.",

        "Spin in circles.",

        "Do nothing useful."
    ]

    # 70% 확률로 좋은 reasoning
    if random.random() < 0.7:

        reasoning = random.choice(good_reasonings)

        reasoning_type = "GOOD"

    # 30% 확률로 나쁜 reasoning
    else:

        reasoning = random.choice(bad_reasonings)

        reasoning_type = "BAD"

    print("\nReasoning Type:")
    print(reasoning_type)

    return reasoning