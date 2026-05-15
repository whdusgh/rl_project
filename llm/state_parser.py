import numpy as np


def parse_observation(obs):

    """
    MiniGrid observation을
    자연어 상태 설명으로 변환
    """

    object_ids = obs[:, :, 0]

    # MiniGrid object IDs
    WALL = 2
    DOOR = 4
    KEY = 5
    GOAL = 8

    descriptions = []

    # key 존재 여부
    if np.any(object_ids == KEY):

        descriptions.append(
            "A key is visible."
        )

    # door 존재 여부
    if np.any(object_ids == DOOR):

        descriptions.append(
            "A door is visible."
        )

    # goal 존재 여부
    if np.any(object_ids == GOAL):

        descriptions.append(
            "The goal is visible."
        )

    # 아무것도 없으면
    if len(descriptions) == 0:

        descriptions.append(
            "The environment is unclear."
        )

    # 하나의 문장으로 합치기
    state_text = " ".join(descriptions)

    return state_text