import numpy as np
import math
import random
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
import matplotlib.pyplot as plt
from typing import TypedDict, Tuple

class Features(TypedDict):
    no_opponent_in_front: bool
    canPass_N: int
    canPass_NE: int
    canPass_E: int
    canPass_SE: int
    canPass_S: int
    canPass_SW: int
    canPass_W: int
    canPass_NW: int
    dist_goal: float
    player_pos: Tuple[float, float]
    team_has_ball: bool
    num_of_opponents_in_player_zone: int
    num_of_teammates_in_player_zone: int
    attempts: int
    score: int
    danger: float
    
feature_dict = {
    "no_opponent_in_front": 0,
    "canPass_N": 1,
    "canPass_NE": 2,
    "canPass_E": 3,
    "canPass_SE": 4,
    "canPass_S": 5,
    "canPass_SW": 6,
    "canPass_W": 7,
    "canPass_NW": 8,
    "dist_goal": 9,
    "player_pos": 10,
    "team_has_ball": 11,
    "num_of_opponents_in_player_zone": 12,
    "num_of_teammates_in_player_zone": 13,
    "attempts": 14,
    "score": 15,
    "danger": 16,
}
    
def features_to_vector(features: Features) -> np.ndarray:
    """
    Converts the features dictionary into a fixed-size vector for DQN training.

    :param features: A dictionary containing the extracted features.
    :return: A NumPy array representing the feature vector.
    """
    vector = [
        int(features["no_opponent_in_front"]),  # Convert boolean to int (0 or 1)
        int(features["canPass_N"]),
        int(features["canPass_NE"]),
        int(features["canPass_E"]),
        int(features["canPass_SE"]),
        int(features["canPass_S"]),
        int(features["canPass_SW"]),
        int(features["canPass_W"]),
        int(features["canPass_NW"]),
        features["dist_goal"],
        features["player_pos"][0],  # x-coordinate
        features["player_pos"][1],  # y-coordinate
        int(features["team_has_ball"]),  # Convert boolean to int (0 or 1)
        features["num_of_opponents_in_player_zone"],
        features["num_of_teammates_in_player_zone"],
        features["attempts"],
        features["score"],
        features["danger"],
    ]
    return np.array(vector, dtype=np.float32)


test = {
    "players": [
        {
            "0": {
                "bbox": [
                    1001.4005737304688,
                    799.7320556640625,
                    1062.396240234375,
                    925.5164794921875,
                ],
                "conf": 0.8071421,
                "position": [1031, 925],
                "position_transformed": [40.253204345703125, 46.666656494140625],
                "team": 1,
                "has_ball": True,
            },
            "3": {
                "bbox": [
                    1039.3731689453125,
                    586.4225463867188,
                    1089.8358154296875,
                    694.4285278320312,
                ],
                "conf": 0.6451548,
                "position": [1064, 694],
                "position_transformed": [40.58443832397461, 37.976226806640625],
                "team": 1,
                # "has_ball": True,
            },
            "2": {
                "bbox": [
                    1039.3731689453125,
                    586.4225463867188,
                    1089.8358154296875,
                    694.4285278320312,
                ],
                "conf": 0.6451548,
                "position": [1064, 694],
                "position_transformed": [43.58443832397461, 37.976226806640625],
                "team": 2,
            },
        },
        {
            "0": {
                "bbox": [
                    1001.4005737304688,
                    799.7320556640625,
                    1062.396240234375,
                    925.5164794921875,
                ],
                "conf": 0.8071421,
                "position": [1031, 925],
                "position_transformed": [47.253204345703125, 46.666656494140625],
                "team": 1,
                "has_ball": True,
            },
            "3": {
                "bbox": [
                    1039.3731689453125,
                    586.4225463867188,
                    1089.8358154296875,
                    694.4285278320312,
                ],
                "conf": 0.6451548,
                "position": [1064, 694],
                "position_transformed": [40.58443832397461, 37.976226806640625],
                "team": 1,
                # "has_ball": True,
            },
            "2": {
                "bbox": [
                    1039.3731689453125,
                    586.4225463867188,
                    1089.8358154296875,
                    694.4285278320312,
                ],
                "conf": 0.6451548,
                "position": [1064, 694],
                "position_transformed": [47.58443832397461, 45.976226806640625],
                "team": 2,
            },
        },
        {
            "0": {
                "bbox": [
                    1001.4005737304688,
                    799.7320556640625,
                    1062.396240234375,
                    925.5164794921875,
                ],
                "conf": 0.8071421,
                "position": [1031, 925],
                "position_transformed": [47.253204345703125, 46.666656494140625],
                "team": 1,
                # "has_ball": True,
            },
            "3": {
                "bbox": [
                    1039.3731689453125,
                    586.4225463867188,
                    1089.8358154296875,
                    694.4285278320312,
                ],
                "conf": 0.6451548,
                "position": [1064, 694],
                "position_transformed": [40.58443832397461, 37.976226806640625],
                "team": 1,
                # "has_ball": True,
            },
            "2": {
                "bbox": [
                    1039.3731689453125,
                    586.4225463867188,
                    1089.8358154296875,
                    694.4285278320312,
                ],
                "conf": 0.6451548,
                "position": [1064, 694],
                "position_transformed": [43.58443832397461, 37.976226806640625],
                "team": 2,
            },
        },
        {
            "5": {
                "bbox": [
                    984.1290283203125,
                    847.8218994140625,
                    1039.0567626953125,
                    997.6185302734375,
                ],
                "conf": 0.48993474,
                "position": [1011, 997],
                "position_transformed": [40.01384735107422, 47.57212829589844],
                "team": 1,
            },
            "6": {
                "bbox": [
                    1053.392578125,
                    604.9873046875,
                    1098.34619140625,
                    763.8143310546875,
                ],
                "conf": 0.45518827,
                "position": [1075, 763],
                "position_transformed": [40.780067443847656, 39.647132873535156],
                "team": 1,
                "has_ball": True,
            },
        },
        {
            "1": {
                "bbox": [
                    975.0267333984375,
                    859.4735107421875,
                    1032.4903564453125,
                    1016.0343017578125,
                ],
                "conf": 0.8010389,
                "position": [1003, 1016],
                "position_transformed": [39.96061706542969, 47.71870422363281],
                "team": 1,
            },
            "5": {
                "bbox": [
                    1050.6195068359375,
                    614.0615234375,
                    1099.0985107421875,
                    769.380859375,
                ],
                "conf": 0.50583255,
                "position": [1074, 769],
                "position_transformed": [40.807369232177734, 39.509281158447266],
                "team": 1,
                "has_ball": True,
            },
        },
        {
            "0": {
                "bbox": [
                    1178.9383544921875,
                    785.50634765625,
                    1260.5804443359375,
                    893.2591552734375,
                ],
                "conf": 0.85533315,
                "position": [1219, 893],
                "position_transformed": [40.54474639892578, 48.03361892700195],
                "team": 2,
            },
            "2": {
                "bbox": [
                    550.6104736328125,
                    915.6600341796875,
                    638.7904052734375,
                    1033.2620849609375,
                ],
                "conf": 0.81979066,
                "position": [594, 1033],
                "position_transformed": [33.73306655883789, 51.6135368347168],
                "team": 1,
            },
            "4": {
                "bbox": [
                    939.5210571289062,
                    569.3636474609375,
                    992.4529418945312,
                    676.5677490234375,
                ],
                "conf": 0.6043398,
                "position": [965, 676],
                "position_transformed": [37.04492950439453, 38.92592239379883],
                "team": 1,
                "has_ball": True,
            },
        },
        {
            "0": {
                "bbox": [
                    532.772216796875,
                    914.7969970703125,
                    646.405029296875,
                    1041.9510498046875,
                ],
                "conf": 0.8584368,
                "position": [589, 1041],
                "position_transformed": [33.653297424316406, 51.586734771728516],
                "team": 1,
            },
            "1": {
                "bbox": [
                    1152.187255859375,
                    789.701416015625,
                    1219.53369140625,
                    903.667724609375,
                ],
                "conf": 0.81976295,
                "position": [1185, 903],
                "position_transformed": [40.04677963256836, 48.01556396484375],
                "team": 2,
            },
            "2": {
                "bbox": [
                    991.7000732421875,
                    564.10693359375,
                    1049.3477783203125,
                    673.718994140625,
                ],
                "conf": 0.7904051,
                "position": [1020, 673],
                "position_transformed": [37.70060348510742, 38.518001556396484],
                "team": 1,
                "has_ball": True,
            },
        },
    ],
    "referees": [],
    "ball": [
        {
            "1": {
                "conf": 0.6297811,
                "bbox": [
                    1058.505126953125,
                    809.33447265625,
                    1089.298583984375,
                    841.92236328125,
                ],
                "position": [1073, 825],
                "position_transformed": [40.75493621826172, 43.268741607666016],
            }
        },
        {
            "1": {
                "conf": 0.5183832,
                "bbox": [
                    1063.429443359375,
                    778.780517578125,
                    1095.869384765625,
                    810.7125244140625,
                ],
                "position": [1079, 794],
                "position_transformed": [40.79155349731445, 41.63777160644531],
            }
        },
        {
            "1": {
                "conf": 0.11,
                "bbox": [
                    1063.429443359375,
                    778.780517578125,
                    1095.869384765625,
                    810.7125244140625,
                ],
                "position": [1079, 794],
                "position_transformed": [40.8347282409668, 41.17273712158203],
            }
        },
        {
            "1": {
                "conf": 0.11,
                "bbox": [
                    1063.429443359375,
                    778.780517578125,
                    1095.869384765625,
                    810.7125244140625,
                ],
                "position": [1079, 794],
                "position_transformed": [40.83425521850586, 40.84413528442383],
            }
        },
        {
            "1": {
                "conf": 0.11,
                "bbox": [
                    1063.429443359375,
                    778.780517578125,
                    1095.869384765625,
                    810.7125244140625,
                ],
                "position": [1079, 794],
                "position_transformed": [40.874351501464844, 40.46513748168945],
            }
        },
        {
            "1": {
                "conf": 0.11,
                "bbox": [
                    1063.429443359375,
                    778.780517578125,
                    1095.869384765625,
                    810.7125244140625,
                ],
                "position": [1079, 794],
                "position_transformed": [38.7821044921875, 44.2365608215332],
            }
        },
        {
            "1": {
                "conf": 0.11,
                "bbox": [
                    1063.429443359375,
                    778.780517578125,
                    1095.869384765625,
                    810.7125244140625,
                ],
                "position": [1079, 794],
                "position_transformed": [38.683048248291016, 43.93621063232422],
            }
        },
    ],
    "goalkeeper": [
        {
            "1": {
                "bbox": [
                    2127.5283203125,
                    742.920166015625,
                    2195.88671875,
                    860.282470703125,
                ],
                "conf": None,
                "position": [2161, 860],
                "position_transformed": [55.04140853881836, 45.69982147216797],
            }
        },
        {},
        {
            "4": {
                "bbox": [
                    2169.1103515625,
                    756.9771728515625,
                    2229.1689453125,
                    890.3621826171875,
                ],
                "conf": None,
                "position": [2199, 890],
                "position_transformed": [55.23343276977539, 45.61698532104492],
            }
        },
        {},
        {
            "4": {
                "bbox": [
                    2184.417724609375,
                    774.5528564453125,
                    2250.401123046875,
                    885.6505126953125,
                ],
                "conf": None,
                "position": [2217, 885],
                "position_transformed": [55.51206588745117, 44.48215866088867],
            }
        },
        {
            "3": {
                "bbox": [
                    2335.18408203125,
                    697.9237670898438,
                    2410.62548828125,
                    805.1900024414062,
                ],
                "conf": None,
                "position": [2372, 805],
                "position_transformed": [56.185935974121094, 46.19419479370117],
            }
        },
        {
            "4": {
                "bbox": [
                    2344.037109375,
                    697.0189208984375,
                    2411.896484375,
                    803.5291748046875,
                ],
                "conf": None,
                "position": [2377, 803],
                "position_transformed": [56.148406982421875, 45.60536575317383],
            }
        },
    ],
    "goalpost": [
        {"1": {"position_transformed": [60, 42], "position": [2434.4, 751.65]}},
        {"1": {"position_transformed": [60, 42], "position": [2450.2, 766.03]}},
        {"1": {"position_transformed": [60, 42], "position": [2468.8, 781.01]}},
        {"1": {"position_transformed": [60, 42], "position": [2484.2, 793.34]}},
        {"1": {"position_transformed": [60, 42], "position": [2494.6, 808.17]}},
        {"1": {"position_transformed": [60, 42], "position": [2517.8, 703.71]}},
        {"1": {"position_transformed": [60, 42], "position": [2540.5, 715.09]}},
    ],
}




FIELD_WIDTH = 50
FIELD_HEIGHT = 90  # haytghyar
BOX_WIDTH = 7.5  # Assuming a fixed width for the boxes
BOX_HEIGHT = 20  # Assuming a fixed height for the boxes
GOAL_POSITION = (105, 34)  # lama yerga3
NUM_PLAYERS = 22  # 11 players per team
NUM_ACTIONS = 17

action_list = {
    0: "pass_N",
    1: "pass_NE",
    2: "pass_E",
    3: "pass_SE",
    4: "pass_S",
    5: "pass_SW",
    6: "pass_W",
    7: "pass_NW",
    8: "dribble_N",
    9: "dribble_NE",
    10: "dribble_E",
    11: "dribble_SE",
    12: "dribble_S",
    13: "dribble_SW",
    14: "dribble_W",
    15: "dribble_NW",
    16: "shoot",
}

action_list_inv = {key: value for value, key in action_list.items()}


boxes = {
    1: {"name": "top_left_box", "top_left_pos": (35, 55)},
    2: {"name": "middle_left_box", "top_left_pos": (35, 35)},
    3: {"name": "bottom_left_box", "top_left_pos": (35, 15)},
    4: {"name": "top_right_box", "top_left_pos": (42.5, 55)},
    5: {"name": "middle_right_box", "top_left_pos": (42.5, 35)},
    6: {"name": "bottom_right_box", "top_left_pos": (42.5, 15)},
}

boxes_danger = {
    "top_left_box": 0,
    "middle_left_box": 0.5,
    "bottom_left_box": 0,
    "top_right_box": 0.8,
    "middle_right_box": 1,
    "bottom_right_box": 0.8,
}


def get_box_danger(box):
    """
    Returns the danger level of a given box.

    :param box: The box to evaluate.
    :return: Danger level (0, 1, or 2).
    """
    if box in boxes_danger:
        return boxes_danger[box]
    else:
        return 0  # Default to 0 if box is not recognized


def get_box_danger_from_position(position):
    """
    Returns the danger level of a box based on the position.

    :param position: The position to evaluate.
    :return: Danger level (0, 1, or 2).
    """
    for box_id, box in boxes.items():
        if (
            box["top_left_pos"][0] <= position[0] <= box["top_left_pos"][0] + BOX_WIDTH
            and box["top_left_pos"][1]
            <= position[1]
            <= box["top_left_pos"][1] + BOX_HEIGHT
        ):
            return boxes_danger[box["name"]]
    return 0  # Default to 0 if no box matches


def get_danger_transition(prev_box, new_box):
    """
    Returns the danger transition level between two boxes.

    :param prev_box: The previous box.
    :param new_box: The new box.
    :return: Danger transition level (0, 1, or 2).
    """
    if prev_box in boxes_danger and new_box in boxes_danger:
        return boxes_danger[new_box] - boxes_danger[prev_box]
    else:
        return 0  # Default to 0 if either box is not recognized


def get_num_of_opponents_in_player_zone(opponents, player_x, player_y):
    """
    Counts the number of opponents in the player's zone.

    :param
    teammates: List of teammate positions.
    :param opponents: List of opponent positions.
    :param player_x: Player's x-coordinate.
    :param player_y: Player's y-coordinate.
    :return: Number of opponents in the player's zone.
    """
    player_zone = [
        (player_x - 5, player_y + 5),  # Top-left corner
        (player_x + 5, player_y - 5),  # Bottom-right corner
    ]

    count = 0
    for opp_x, opp_y in opponents:
        if (
            player_zone[0][0] <= opp_x <= player_zone[1][0]
            and player_zone[0][1] >= opp_y >= player_zone[1][1]
        ):
            count += 1

    return count


def get_num_of_teammates_in_player_zone(teammates, player_x, player_y):
    """
    Counts the number of teammates in the player's zone.

    :param teammates: List of teammate positions.
    :param player_x: Player's x-coordinate.
    :param player_y: Player's y-coordinate.
    :return: Number of teammates in the player's zone.
    """
    player_zone = [
        (player_x - 5, player_y + 5),  # Top-left corner
        (player_x + 5, player_y - 5),  # Bottom-right corner
    ]

    count = 0
    for mate_x, mate_y in teammates:
        if (
            player_zone[0][0] <= mate_x <= player_zone[1][0]
            and player_zone[0][1] >= mate_y >= player_zone[1][1]
        ):
            count += 1

    return count


def compute_reward(state, action, next_state):
    reward = 0.0
    print("\n--- Compute Reward Debug ---")
    print(f"State: {state}")
    print(f"Action: {action} ({action_list[action]})")
    print(f"Next State: {next_state}")

    # Basic event-based rewards
    if next_state["score"] - state["score"] > 700:
        reward += 1.0
        print("Goal scored! Reward: +1.0")
    elif next_state["score"] - state["score"] > 0:
        reward += 0.3
        print("Shot on target! Reward: +0.3")
    elif action_list[action] != "shoot" and next_state["team_has_ball"]:
        reward += 0.05
        print("Team still has the ball. Reward: +0.05")
    elif not next_state["team_has_ball"] or (
        (next_state["score"] == state["score"])
        and next_state["attempts"] != state["attempts"]
    ):
        reward -= 0.4
        print("Turnover occurred. Penalty: -0.4")

    # Progress toward goal
    if next_state["team_has_ball"]:
        distance_reward = (state["dist_goal"] - next_state["dist_goal"]) * 0.05
        reward += distance_reward
        print(f"Progress toward goal. Distance reward: {distance_reward:.2f}")

    # Encourage faster decisions
    reward -= 0.01
    print("Time penalty applied. Penalty: -0.01")

    # Spatial danger transition reward
    prev_danger = state["danger"]
    if next_state["team_has_ball"]:
        new_danger = next_state["danger"]
        danger_delta = new_danger - prev_danger
        reward += danger_delta * 0.2
        print(
            f"Danger transition. Danger delta: {danger_delta:.2f}, Reward: {danger_delta * 0.2:.2f}"
        )

    # Action-based encouragement based on danger
    if "shoot" in action_list[action]:
        if prev_danger >= 0.8:
            reward += 0.3
            print("Shooting in high-danger zone. Reward: +0.3")
        else:
            reward -= 0.2
            print("Shooting in low-danger zone. Penalty: -0.2")
    else:
        new_danger = next_state["danger"]
        if new_danger < 0.8:
            reward += 0.2
            print("Safe progression in low-danger zone. Reward: +0.2")
        else:
            reward += 0.05
            print("Progression in high-danger zone. Reward: +0.05")

    # Zone congestion and support
    num_teammates = next_state["num_of_teammates_in_player_zone"]
    num_opponents = next_state["num_of_opponents_in_player_zone"]

    penalty = 0.2 * (num_opponents)
    reward -= penalty
    print(f"Zone congestion. Opponents: {num_opponents}, Penalty: -{penalty:.2f}")

    reward += 0.1 * num_teammates
    print(
        f"Teammates in zone. Teammates: {num_teammates}, Reward: +{0.1 * num_teammates:.2f}"
    )

    # Count available pass directions
    pass_option_count = (
        next_state["canPass_N"]
        + next_state["canPass_NE"]
        + next_state["canPass_E"]
        + next_state["canPass_SE"]
        + next_state["canPass_S"]
        + next_state["canPass_SW"]
        + next_state["canPass_W"]
        + next_state["canPass_NW"]
    )
    reward += 0.05 * pass_option_count
    print(
        f"Pass options available. Count: {pass_option_count}, Reward: +{0.05 * pass_option_count:.2f}"
    )

    if not next_state["no_opponent_in_front"]:
        if pass_option_count > 0:
            bonus = 0.1 * pass_option_count * (1 + new_danger)
            reward += bonus
            print(f"Opponent in front but pass options available. Bonus: +{bonus:.2f}")
        else:
            penalty = 0.2 * (1 + new_danger)
            reward -= penalty
            print(f"No pass options and under pressure. Penalty: -{penalty:.2f}")

    if state["no_opponent_in_front"]:
        if "dribble" in action_list[action] and "E" in action_list[action]:
            reward += 0.3
            print("Aggressive play with no opponent in front. Reward: +0.3")
        elif "pass" in action_list[action]:
            reward -= 0.1
            print("Unnecessary pass with no opponent in front. Penalty: -0.1")

    if "pass" in action_list[action]:
        pass_directions = {
            "N": state["canPass_N"],
            "NE": state["canPass_NE"],
            "E": state["canPass_E"],
            "SE": state["canPass_SE"],
            "S": state["canPass_S"],
            "SW": state["canPass_SW"],
            "W": state["canPass_W"],
            "NW": state["canPass_NW"],
        }

        pass_direction = action_list[action].split("_")[1]
        if pass_directions.get(pass_direction, 0) > 0:
            reward += 0.1
            print(f"Valid pass direction ({pass_direction}). Reward: +0.1")
        else:
            reward -= 0.1
            print(f"Invalid pass direction ({pass_direction}). Penalty: -0.1")

    print(f"Total Reward: {reward:.2f}")
    return reward


def count_points_in_rectangle(teammates, opponents, rect_top_left, rect_bottom_right):
    """
    Counts how many points are inside a given rectangle.

    :param points: List of tuples representing points [(x1, y1), (x2, y2), ...].
    :param rect_top_left: Tuple (x, y) representing the top-left corner of the rectangle.
    :param rect_bottom_right: Tuple (x, y) representing the bottom-right corner of the rectangle.
    :return: Integer count of points inside the rectangle.
    """
    count = 0
    for x, y in teammates:
        if (
            rect_top_left[0] <= x <= rect_bottom_right[0]
            and rect_top_left[1] >= y >= rect_bottom_right[1]
        ):
            count += 1
    for x, y in opponents:
        if (
            rect_top_left[0] <= x <= rect_bottom_right[0]
            and rect_top_left[1] >= y >= rect_bottom_right[1]
        ):
            count -= 1
    return count


def count_agent_in_rectangle(agent_x, agent_y):
    """
    Counts how many points are inside a given rectangle.

    :param points: List of tuples representing points [(x1, y1), (x2, y2), ...].
    :param rect_top_left: Tuple (x, y) representing the top-left corner of the rectangle.
    :param rect_bottom_right: Tuple (x, y) representing the bottom-right corner of the rectangle.
    :return: Integer count of points inside the rectangle.
    """
    top_rect = [(agent_x, 65), (50, 51)]
    middle_rect = [(agent_x, 51), (50, 39)]
    bottom_rect = [(agent_x, 39), (50, 25)]
    count = 0
    if (
        top_rect[0][0] <= agent_x <= top_rect[1][0]
        and top_rect[0][1] >= agent_y >= top_rect[1][1]
    ):
        count = 1
    if (
        middle_rect[0][0] <= agent_x <= middle_rect[1][0]
        and middle_rect[0][1] >= agent_y >= middle_rect[1][1]
    ):
        count = 2
    if (
        bottom_rect[0][0] <= agent_x <= bottom_rect[1][0]
        and bottom_rect[0][1] >= agent_y >= bottom_rect[1][1]
    ):
        count = 3
    return count


class SoccerFeatureExtractor:
    @staticmethod
    def get_distance(x1, y1, x2, y2):
        return math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)

    @staticmethod
    def get_angle(x1, y1, x2, y2):
        dx = x2 - x1
        dy = y2 - y1
        return math.degrees(math.atan2(dy, dx))

    ################################################################### features

    # @staticmethod
    # def get_pass_directions(player_pos, teammates, opponents):
    #     directions = ["N", "NE", "E", "SE", "S", "SW", "W", "NW"]
    #     direction_flags = {dir: 0 for dir in directions}

    #     all_players = [(pos, "teammate") for pos in teammates] + [
    #         (pos, "opponent") for pos in opponents
    #     ]
    #     all_players_with_ids = {
    #         player_id: player for player_id, player in enumerate(all_players)
    #     }
    #     opponent_last_x = max([pos[0] for pos in opponents]) if opponents else 0
    #     print("opponent_last_x", all_players_with_ids)

    #     def get_angle(dx, dy):
    #         angle = math.degrees(math.atan2(dy, dx)) % 360
    #         return angle

    #     direction_ranges = {
    #         "E": (-22.5, 22.5),
    #         "NE": (22.5, 67.5),
    #         "N": (67.5, 112.5),
    #         "NW": (112.5, 157.5),
    #         "W": (157.5, 202.5),
    #         "SW": (202.5, 247.5),
    #         "S": (247.5, 292.5),
    #         "SE": (292.5, 337.5),
    #     }

    #     # Catch wraparound for E (0 to 360)
    #     direction_ranges["E"] = (-22.5 % 360, 22.5)

    #     # Dictionary to hold closest player per direction
    #     closest = {
    #         dir: (None, None, float("inf")) for dir in directions
    #     }  # (type,id, distance)
    #     print("clositemssest", all_players_with_ids.items())

    #     for player_id, (pos, player_type) in all_players_with_ids.items():
    #         dx = pos[0] - player_pos[0]
    #         dy = pos[1] - player_pos[1]
    #         dist = math.hypot(dx, dy)
    #         angle = get_angle(dx, dy)
    #         print(angle, "mido angle")

    #         for dir in directions:
    #             low, high = direction_ranges[dir]
    #             # Handle wraparound for E
    #             if low > high:
    #                 in_range = angle >= low or angle <= high
    #             else:
    #                 in_range = low <= angle <= high

    #             if in_range and dist < closest[dir][2]:
    #                 closest[dir] = (player_type, player_id, dist)

    #     print("closest2", closest)

    #     for dir in directions:
    #         if closest[dir][1] is not None and closest[dir][1] in all_players_with_ids:
    #             print(
    #                 "all_players_with_ids[closest[dir][1]][0][0]",
    #                 all_players_with_ids[closest[dir][1]][0][0],
    #             )

    #             if (
    #                 closest[dir][0] == "teammate"
    #                 and all_players_with_ids[closest[dir][1]][0][0] < opponent_last_x
    #             ):
    #                 print( all_players_with_ids[closest[dir][1]],'midoo')
    #                 direction_flags[dir] = 1

    #     print("Direction flags:", direction_flags)
    #     plot_pitch(teammates, opponents, player_pos)

    #     return direction_flags

    @staticmethod
    def get_opponent_and_pass_data(agent_x, agent_y, teammates, opponents) -> dict[
        str,
        bool,
    ]:

        # Initialize closest opponent variablesF
        no_opponent_in_front = True
        last_opp_x = 0

        # Initialize 8-direction tracking (will only store opponents)
        directions = ["N", "NE", "E", "SE", "S", "SW", "W", "NW"]
        dir_data = {d: {"dist": float("inf"), "pos": None} for d in directions}

        # Temporary storage for teammate positions and distances
        team_data = {d: {"dist": float("inf"), "pos": None} for d in directions}
        pass_directions = {
            d: 0 for d in directions
        }  # 1 if teammate is closest, 0 otherwise

        # Process teammates first (store temporarily)
        for mate_x, mate_y in teammates:
            dx = mate_x - agent_x
            dy = mate_y - agent_y
            dist = math.hypot(dx, dy)
            angle = math.degrees(math.atan2(dy, dx))

            if -22.5 <= angle < 22.5:
                direction = "E"
            elif 22.5 <= angle < 67.5:
                direction = "NE"
            elif 67.5 <= angle < 112.5:
                direction = "N"
            elif 112.5 <= angle < 157.5:
                direction = "NW"
            elif angle >= 157.5 or angle < -157.5:
                direction = "W"
            elif -157.5 <= angle < -112.5:
                direction = "SW"
            elif -112.5 <= angle < -67.5:
                direction = "S"
            elif -67.5 <= angle < -22.5:
                direction = "SE"

            # Update team data if this is the closest teammate in this direction
            if dist < team_data[direction]["dist"]:
                team_data[direction] = {"dist": dist, "pos": (mate_x, mate_y)}

        # Process opponents (store in dir_data)
        if opponents:
            min_dist = float("inf")
            all_opp_behind = True

            for opp_x, opp_y in opponents:
                # Check if opponent is in front
                if opp_x > last_opp_x:
                    last_opp_x = opp_x

                if agent_x <= opp_x and agent_y - 2.5 < opp_y < agent_y + 2.5:
                    all_opp_behind = False

                dist = math.hypot(opp_x - agent_x, opp_y - agent_y)
                # print(opp_x, opp_y, agent_x, agent_y, dist)

                # Track closest opponent overall
                if dist < min_dist:
                    min_dist = dist
                    closest_opp = (opp_x, opp_y)

                # Classify direction
                dx = opp_x - agent_x
                dy = opp_y - agent_y
                angle = math.degrees(math.atan2(dy, dx))

                if -22.5 <= angle < 22.5:
                    direction = "E"
                elif 22.5 <= angle < 67.5:
                    direction = "NE"
                elif 67.5 <= angle < 112.5:
                    direction = "N"
                elif 112.5 <= angle < 157.5:
                    direction = "NW"
                elif angle >= 157.5 or angle < -157.5:
                    direction = "W"
                elif -157.5 <= angle < -112.5:
                    direction = "SW"
                elif -112.5 <= angle < -67.5:
                    direction = "S"
                elif -67.5 <= angle < -22.5:
                    direction = "SE"

                # Update opponent data if this is the closest opponent in this direction
                if dist < dir_data[direction]["dist"]:
                    dir_data[direction] = {"dist": dist, "pos": (opp_x, opp_y)}

            # Set opponent metrics

            no_opponent_in_front = all_opp_behind

        # Determine pass directions by comparing team_data and dir_data

        ### checck here for offside rule ba3dennnnnnnnnn mesh ma3mola

        for direction in directions:
            angle = 90
            team_dist = team_data[direction]["dist"]
            opp_dist = dir_data[direction]["dist"]
            opp_x, opp_y = (
                dir_data[direction]["pos"] if dir_data[direction]["pos"] else (0, 0)
            )
            mate_x, mate_y = (
                team_data[direction]["pos"] if team_data[direction]["pos"] else (0, 0)
            )
            if team_dist != float("inf") and opp_dist != float("inf"):
                dx = mate_x - agent_x
                dy = mate_y - agent_y
                angle_mate = math.degrees(math.atan2(dy, dx))
                dx = opp_x - agent_x
                dy = opp_y - agent_y
                angle_opp = math.degrees(math.atan2(dy, dx))
                angle = abs(angle_mate - angle_opp)

            # If there's a teammate and either no opponent or teammate is closer
            if (
                team_dist != float("inf")
                and (mate_x < last_opp_x or (no_opponent_in_front and mate_x < agent_x))
                and angle > 10
            ):
                pass_directions[direction] = 1

        # Prepare return values (dir_data always contains opponent positions)
        return {
            "no_opponent_in_front": no_opponent_in_front,
            "canPass_N": pass_directions["N"],
            "canPass_NE": pass_directions["NE"],
            "canPass_E": pass_directions["E"],
            "canPass_SE": pass_directions["SE"],
            "canPass_S": pass_directions["S"],
            "canPass_SW": pass_directions["SW"],
            "canPass_W": pass_directions["W"],
            "canPass_NW": pass_directions["NW"],
        }

    @staticmethod
    def extract_features(
        teammates,
        opponents,
        attempts,
        score,
        goalpost,
        goalkeeper,
        agent_x=0,
        agent_y=0,
    )-> Features:

        # lama nozbot el points

        # max_dist = math.sqrt(FIELD_WIDTH**2 + FIELD_HEIGHT**2)
        dist_goal = 100
        if goalpost:
            goalpost[1] = FIELD_HEIGHT - goalpost[1]
            dist_goal = SoccerFeatureExtractor.get_distance(
                agent_x, agent_y, goalpost[0], goalpost[1]
            )

        features: Features = SoccerFeatureExtractor.get_opponent_and_pass_data(
            agent_x, agent_y, teammates, opponents
        )
        features["dist_goal"] = dist_goal
        features["player_pos"] = (agent_x, agent_y)
        features["team_has_ball"] = True if agent_x > 0 else False
        num_opponents = get_num_of_opponents_in_player_zone(opponents, agent_x, agent_y)
        num_teammates = get_num_of_teammates_in_player_zone(teammates, agent_x, agent_y)
        features["num_of_opponents_in_player_zone"] = num_opponents
        features["num_of_teammates_in_player_zone"] = num_teammates
        features["attempts"] = attempts
        features["score"] = score
        features["danger"] = get_box_danger_from_position((agent_x, agent_y))
        print("features", features)
       
        return features

       


######################################################################################################


def plot_pitch(teammates, opponets, me):
    plt.figure(figsize=(10, 6))
    plt.xlim(0, 50)
    plt.ylim(0, 90)
    plt.title("Soccer Field")
    plt.xlabel("Width")
    plt.ylabel("Height")
    for box_id, box_info in boxes.items():
        top_left = box_info["top_left_pos"]
        width = BOX_WIDTH  # Assuming a fixed width for the boxes
        height = BOX_HEIGHT  # Assuming a fixed height for the boxes
        plt.gca().add_patch(
            plt.Rectangle(
                top_left,
                width,
                height,
                color={
                    "top_left_box": "red",
                    "middle_left_box": "orange",
                    "bottom_left_box": "yellow",
                    "top_right_box": "green",
                    "middle_right_box": "blue",
                    "bottom_right_box": "purple",
                }.get(
                    box_info["name"], "gray"
                ),  # Default to gray if name not found
                alpha=0.5,
                label=box_info["name"],
            )
        )

        # Draw rectangles
    # plt.gca().add_patch(plt.Rectangle(
    #     top_rect[0],  # Top-left corner
    #     top_rect[1][0] - top_rect[0][0],  # Width (x2 - x1)
    #     top_rect[1][1] - top_rect[0][1],  # Height (y2 - y1)
    #     color='red', alpha=0.5, label='Top Rectangle'
    # ))
    # plt.gca().add_patch(plt.Rectangle(
    #     middle_rect[0],
    #     middle_rect[1][0] - middle_rect[0][0],
    #     middle_rect[1][1] - middle_rect[0][1],
    #     color='blue', alpha=0.5, label='Middle Rectangle'
    # ))
    # plt.gca().add_patch(plt.Rectangle(
    #     bottom_rect[0],
    #     bottom_rect[1][0] - bottom_rect[0][0],
    #     bottom_rect[1][1] - bottom_rect[0][1],
    #     color='green', alpha=0.5, label='Bottom Rectangle'
    # ))

    plt.scatter(me[0], me[1], color="blue", label="Agent")
    for mate_x, mate_y in teammates:
        plt.scatter(mate_x, mate_y, color="yellow", label="Teammate")

    # Plot opponents
    for opp_x, opp_y in opponets:
        plt.scatter(opp_x, opp_y, color="red", label="Opponent")

    # for printing scene
    plt.grid()
    plt.show()


def get_state_from_positions(scene):
    """Convert raw positions to state for specific player"""
    teammates = []
    opponents = []
    player_pos = (0, 0)
    attempts = scene["Attempts"] if scene.get("Attempts", False) else 0
    score = scene["Score"] if scene.get("Score", False) else 0
    goalpost = None
    goalkeeper = None
    if scene.get("goalpost", False):
        goalpost = scene["goalpost"][list(scene["goalpost"].keys())[0]][
            "position_transformed"
        ]
    if scene.get("goalkeeper", False):
        first_key = next(iter(scene["goalkeeper"]))
        x, y = scene["goalkeeper"][first_key]["position_transformed"]
        goalkeeper = (x, FIELD_HEIGHT - y)
        # print("goalkeeper",scene['goalkeeper'][first_key]['position_transformed'])
    if scene.get("players", False):
        for player_id, player_data in scene["players"].items():
            x, y = player_data["position_transformed"]
            if player_data["team"] == 1:
                if player_data.get("has_ball", False):
                    player_pos = (x, FIELD_HEIGHT - y)
                else:
                    teammates.append((x, FIELD_HEIGHT - y))
            else:
                opponents.append((x, FIELD_HEIGHT - y))

    # plot_pitch(teammates, opponents, player_pos)

    return SoccerFeatureExtractor.extract_features(
        teammates,
        opponents,
        attempts,
        score,
        goalpost,
        goalkeeper,
        player_pos[0],
        player_pos[1],
    )


def calculate_reward(prev_state, new_state, action):
    """
    Normalized reward function that keeps values in a reasonable range (-10 to +10)
    """
    reward = 0

    # Distance to goal improvement (normalize from -5 to +5)
    if (
        prev_state[feature_dict["dist_goal"]] != 0
        and new_state[feature_dict["dist_goal"]] != 0
    ):
        # Scale based on field size, assuming max dist ~60
        dist_change = (
            prev_state[feature_dict["dist_goal"]] - new_state[feature_dict["dist_goal"]]
        )
        normalized_dist_change = max(min(dist_change, 5), -5)  # Clamp between -5 and 5
        reward += normalized_dist_change

    # Facing goal (normalized to +1)
    if (
        abs(prev_state[feature_dict["angle_goal"]])
        - abs(new_state[feature_dict["angle_goal"]])
    ) > 0:
        reward += 1

    # Clear path vs opponent in front (normalized to +3/-1)
    reward += 3 if new_state[feature_dict["no_opponent_in_front"]] else -0.1

    # Goal visibility rewards/penalties (normalized to +2/-2)
    if (
        not new_state[feature_dict["goaldetected"]]
        and not prev_state[feature_dict["goaldetected"]]
    ):
        reward += -2  # Still can't see goal
    elif (
        new_state[feature_dict["goaldetected"]]
        and not prev_state[feature_dict["goaldetected"]]
    ):
        reward += 2  # Now can see goal
    elif (
        new_state[feature_dict["goaldetected"]]
        and prev_state[feature_dict["goaldetected"]]
    ):
        reward += 0.1  # Still can see goal
    elif (
        not new_state[feature_dict["goaldetected"]]
        and prev_state[feature_dict["goaldetected"]]
    ):
        reward += -2  # Lost sight of goal

    # Time penalty (small negative reward for each step in same attack)
    if prev_state[feature_dict["attempts"]] == new_state[feature_dict["attempts"]]:
        reward -= 0.2

    # Goal scored (maximum reward of +10)
    if new_state[feature_dict["score"]] - prev_state[feature_dict["score"]] > 700:
        reward += 100  # Goal scored
        return 100
    elif new_state[feature_dict["score"]] - prev_state[feature_dict["score"]] > 0:
        reward += 5  # Shot on target but missed

    # Turnover (normalized to -5)
    if (
        new_state[feature_dict["score"]] == prev_state[feature_dict["score"]]
    ) and new_state[feature_dict["attempts"]] != prev_state[feature_dict["attempts"]]:
        # reward -= 5   # Turnover/lost possession
        if action == 16:
            reward -= 1
            return -2
        else:
            reward -= 10
            return -10

    # Pass reward/penalty based on pass direction flags
    # Check if action was a pass (actions 0-7 correspond to pass_N_flag through pass_NW_flag)
    if 0 <= action <= 7:
        # Map action to direction index
        direction_idx = (
            action - 7 + feature_dict["pass_N_flag"]
        )  # Calculate the feature index

        # Severe penalty for passing when no teammate available
        if prev_state[direction_idx] == 0:
            reward -= 8  # Bad pass decision (normalized from -1000)
        # Reward for successful pass to open teammate
        elif prev_state[direction_idx] == 1:
            reward += 2  # Good pass (normalized from +10)

    # Congestion change reward (assess how crowded the area around player has become)
    prev_cong_num = prev_state[feature_dict["agent_box"]]
    new_cong_num = new_state[feature_dict["agent_box"]]
    print(prev_cong_num)
    print(new_cong_num)

    if new_cong_num == 0:
        reward -= 1
    else:
        prev_congestion = prev_state[feature_dict[boxes[prev_cong_num]["name"]]]
        new_congestion = new_state[feature_dict[boxes[new_cong_num]["name"]]]
        if new_congestion > prev_congestion:
            reward += 1
        elif new_congestion < prev_congestion:
            reward -= 1

    return reward


# DQn classs


class DQNAgent:
    def __init__(self, load_path=None):
        """
        Initialize DQN Agent.

        Args:
            state_size (int): Size of the state space
            action_size (int): Number of possible actions
            load_path (str, optional): Path to load pre-trained model. If None, creates new model.
        """
        self.state_size = len(feature_dict) + 1
        self.action_size = len(action_list)
        self.memory = deque(maxlen=10000)
        self.gamma = 0.95
        self.epsilon = (
            1.0 if load_path is None else 0.01
        )  # Start with low epsilon if loading
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.learning_rate = 0.001
        self.model = self._build_model()
        self.target_model = self._build_model()
        self.prev_state = None

        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)

        if load_path is not None:
            self.load(load_path)
        else:
            self.update_target_model()

    def _build_model(self):
        model = nn.Sequential(
            nn.Linear(self.state_size, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, self.action_size),
        )
        return model

    def update_target_model(self):
        self.target_model.load_state_dict(self.model.state_dict())

    def remember(self, state, action, reward, next_state, done):
        stateVector = features_to_vector(state)
        next_stateVector = features_to_vector(next_state)
        self.memory.append((stateVector, action, reward, next_stateVector, done))

    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        stateVector = features_to_vector(state)
        stateVector = torch.FloatTensor(stateVector).unsqueeze(0)
        act_values = self.model(stateVector)
        return torch.argmax(act_values).item()

    def replay(self, batch_size):
        if len(self.memory) <= batch_size:
            return

        minibatch = random.sample(self.memory, batch_size)
        states = torch.FloatTensor(np.array([t[0] for t in minibatch]))
        actions = torch.LongTensor(np.array([t[1] for t in minibatch]))
        rewards = torch.FloatTensor(np.array([t[2] for t in minibatch]))
        next_states = torch.FloatTensor(np.array([t[3] for t in minibatch]))
        dones = torch.FloatTensor(np.array([t[4] for t in minibatch]))

        current_q = self.model(states).gather(1, actions.unsqueeze(1))
        next_q = self.target_model(next_states).max(1)[0].detach()
        target = rewards + (1 - dones) * self.gamma * next_q

        loss = nn.MSELoss()(current_q.squeeze(), target)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def save(self, filename, log_filename="model_log.txt"):
        """Save the model and training parameters, and log them to a text file."""
        # Save the model and training parameters
        torch.save(
            {
                "model_state_dict": self.model.state_dict(),
                "target_model_state_dict": self.target_model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "memory": list(self.memory),
                "epsilon": self.epsilon,
            },
            filename,
        )

        # Log parameters to a text file
        with open(log_filename, "w") as log_file:
            log_file.write("Model Weights:\n")
            for name, param in self.model.named_parameters():
                log_file.write(f"{name}:\n{param.data}\n\n")

            log_file.write("\nTarget Model Weights:\n")
            for name, param in self.target_model.named_parameters():
                log_file.write(f"{name}:\n{param.data}\n\n")

            log_file.write("\nReplay Memory:\n")
            for i, memory in enumerate(self.memory):
                log_file.write(f"Memory {i}: {memory}\n")

            log_file.write(f"\nEpsilon: {self.epsilon}\n")

    def load(self, filename):
        """Load the model and training parameters."""
        checkpoint = torch.load(filename)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.target_model.load_state_dict(checkpoint["target_model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.memory = deque(checkpoint["memory"], maxlen=10000)
        self.epsilon = checkpoint["epsilon"]

    def get_action(self, scene):
        """
        Get the best action for a given game scene.

        Args:
            scene (dict): Game scene containing player positions, ball possession, etc.

        Returns:
            str: The chosen action from the action_list dictionary
        """
        # Convert scene to state
        state = get_state_from_positions(scene)
        # Convert state to vector
        state_vector = features_to_vector(state)

        # Get action using the trained model
        self.model.eval()  # Set model to evaluation mode
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state_vector).unsqueeze(0)
            action_values = self.model(state_tensor)
            action_idx = torch.argmax(action_values).item()

        # Return the corresponding action string
        return action_list[action_idx]


if __name__ == "__main__":
    # rl_agent = DQNAgent(load_path="trained_model.pth")
    # rl_agent.save("trained_model.pth", "model_logbefore.txt")

    # first_time = True
    # prev_action = None
    # trainNN_counter = 0
    scene1 = {
        "players": test["players"][0],
        "ball": test["ball"][0],
        "goalkeeper": test["goalkeeper"][0],
        "goalpost": test["goalpost"][0],
        "referees": test["referees"],
        "Score": 0,
    }
    scene2 = {
        "players": test["players"][1],
        "ball": test["ball"][1],
        "goalkeeper": test["goalkeeper"][1],
        "goalpost": test["goalpost"][1],
        "referees": test["referees"],
        "Score": 0,
    }
    scene3 = {
        "players": test["players"][2],
        "ball": test["ball"][2],
        "goalkeeper": test["goalkeeper"][2],
        "goalpost": test["goalpost"][2],
        "referees": test["referees"],
        "Score": 800,
    }

    state = get_state_from_positions(scene1)
    next_state = get_state_from_positions(scene2)
    next_state2 = get_state_from_positions(scene3)
    action = action_list_inv["dribble_E"]
    reward = compute_reward(state, action, next_state)
    shoot = action_list_inv["shoot"]
    reward_shoot = compute_reward(state, shoot, next_state2)
    # print("Initial State:", state)
    # print("Next State:", next_state)
    print("Reward:", reward)
    print("Reward shoot:", reward_shoot)
    # while True:
    #     cur_state = [random.randint(1, 3) for _ in range(28)]  # Random integers between 1 and 3

    #     prev_state = rl_agent.prev_state
    #     action_taken = None

    #     if first_time:
    #         first_time = False
    #         action_taken = rl_agent.act(state=cur_state)
    #         rl_agent.prev_state = cur_state
    #         prev_action = action_taken
    #     else:
    #         action_taken = rl_agent.act(state=cur_state)
    #         print("action",action_taken)
    #         prev_action = action_taken
    #         reward = calculate_reward(prev_state, cur_state, prev_action)
    #         print("Reward:", reward)
    #         rl_agent.remember(prev_state, prev_action, reward, cur_state, done=False)

    #     trainNN_counter += 1
    #     if trainNN_counter == 64:
    #         print("Training the model...")
    #         rl_agent.replay(50)
    #         trainNN_counter = 0

    #         # Save and log parameters
    #         rl_agent.save("trained_model.pth", "model_logafter.txt")
    #         break

    #     prev_action = action_taken
