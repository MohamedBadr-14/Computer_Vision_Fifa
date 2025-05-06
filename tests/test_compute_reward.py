import unittest
from rl import compute_reward, feature_dict, get_state_from_positions, plot_pitch

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
                "position_transformed": [38.58443832397461, 35.976226806640625],
                "team": 1,
            },
            "1": {
                "bbox": [
                    1039.3731689453125,
                    586.4225463867188,
                    1089.8358154296875,
                    694.4285278320312,
                ],
                "conf": 0.6451548,
                "position": [1064, 694],
                "position_transformed": [45.8443832397461, 35.976226806640625],
                "team": 1,
            },
            "9": {
                "bbox": [
                    1039.3731689453125,
                    586.4225463867188,
                    1089.8358154296875,
                    694.4285278320312,
                ],
                "conf": 0.6451548,
                "position": [1064, 694],
                "position_transformed": [20.58443832397461, 80.976226806640625],
                "team": 1,
            },
            "7": {
                "bbox": [
                    1039.3731689453125,
                    586.4225463867188,
                    1089.8358154296875,
                    694.4285278320312,
                ],
                "conf": 0.6451548,
                "position": [1064, 694],
                "position_transformed": [35.8443832397461, 55.976226806640625],
                "team": 1,
            },
            "10": {
                "bbox": [
                    1039.3731689453125,
                    586.4225463867188,
                    1089.8358154296875,
                    694.4285278320312,
                ],
                "conf": 0.6451548,
                "position": [1064, 694],
                "position_transformed": [38.8443832397461, 55.976226806640625],
                "team": 2,
            },
            "11": {
                "bbox": [
                    1039.3731689453125,
                    586.4225463867188,
                    1089.8358154296875,
                    694.4285278320312,
                ],
                "conf": 0.6451548,
                "position": [1064, 694],
                "position_transformed": [44.8443832397461, 55.976226806640625],
                "team": 1,
                "has_ball": True,
            },
        },
        {
            "0": {
                "bbox": [
                    1000.3087158203125,
                    819.2086181640625,
                    1054.7247314453125,
                    953.2657470703125,
                ],
                "conf": 0.7896831,
                "position": [1027, 953],
                "position_transformed": [40.17107391357422, 47.09035873413086],
                "team": 1,
            },
            "6": {
                "bbox": [
                    1037.8121337890625,
                    591.7083740234375,
                    1087.9847412109375,
                    714.015869140625,
                ],
                "conf": 0.39994213,
                "position": [1062, 714],
                "position_transformed": [40.53303909301758, 38.3753662109375],
                "team": 1,
                "has_ball": True,
            },
        },
        {
            "0": {
                "bbox": [
                    989.3291015625,
                    827.08544921875,
                    1049.1044921875,
                    978.5794677734375,
                ],
                "conf": 0.806077,
                "position": [1019, 978],
                "position_transformed": [40.107818603515625, 47.37126159667969],
                "team": 1,
            },
            "5": {
                "bbox": [
                    1046.9359130859375,
                    602.583740234375,
                    1095.2481689453125,
                    736.2244873046875,
                ],
                "conf": 0.5141418,
                "position": [1071, 736],
                "position_transformed": [40.72050476074219, 38.871307373046875],
                "team": 1,
                "has_ball": True,
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


class TestComputeReward(unittest.TestCase):
    def setUp(self):
        # Set up common states for testing
        self.scene1 = {
            "players": test["players"][0],
            "ball": test["ball"][0],
            "goalkeeper": test["goalkeeper"][0],
            "goalpost": test["goalpost"][0],
            "referees": test["referees"],
        }
        self.scene2 = {
            "players": test["players"][1],
            "ball": test["ball"][1],
            "goalkeeper": test["goalkeeper"][1],
            "goalpost": test["goalpost"][1],
            "referees": test["referees"],
        }
        self.next_state = self.state.copy()
        self.action = 16  # "shoot"

    def test_goal_scored(self):
        state = get_state_from_positions(self.scene1)
        
        reward = compute_reward(self.state, self.action, self.next_state)
        self.assertEqual(reward, 1.0, "Reward should be 1.0 for scoring a goal.")

    def test_on_target(self):
        self.next_state[feature_dict["score"]] = 1
        reward = compute_reward(self.state, self.action, self.next_state)
        self.assertEqual(reward, 0.3, "Reward should be 0.3 for a shot on target.")

    def test_turnover(self):
        self.next_state[feature_dict["attempts"]] = 1
        self.next_state["opponent_has_ball"] = True
        reward = compute_reward(self.state, self.action, self.next_state)
        self.assertEqual(reward, -0.4, "Reward should be -0.4 for a turnover.")

    def test_progress_toward_goal(self):
        self.next_state["distance_to_goal"] = 40
        reward = compute_reward(self.state, self.action, self.next_state)
        self.assertAlmostEqual(
            reward,
            0.5,
            delta=0.01,
            msg="Reward should reflect progress toward the goal.",
        )

    def test_danger_transition(self):
        self.next_state["danger"] = 0.7
        reward = compute_reward(self.state, self.action, self.next_state)
        self.assertAlmostEqual(
            reward, 0.04, delta=0.01, msg="Reward should reflect danger transition."
        )

    def test_zone_congestion(self):
        self.next_state["num_of_opponents_in_player_zone"] = 4
        self.next_state["num_of_teammates_in_player_zone"] = 3
        reward = compute_reward(self.state, self.action, self.next_state)
        self.assertAlmostEqual(
            reward, -0.1, delta=0.01, msg="Reward should reflect zone congestion."
        )

    def test_pass_directions(self):
        self.action = 0  # "pass_N"
        reward = compute_reward(self.state, self.action, self.next_state)
        self.assertAlmostEqual(
            reward, 0.1, delta=0.01, msg="Reward should reflect valid pass direction."
        )

    def test_invalid_pass(self):
        self.action = 1  # "pass_NE"
        reward = compute_reward(self.state, self.action, self.next_state)
        self.assertAlmostEqual(
            reward,
            -0.1,
            delta=0.01,
            msg="Reward should penalize invalid pass direction.",
        )


if __name__ == "__main__":
    unittest.main()
