from policy.behavior_policy.epsilon_greedy import EpsilonGreedy

import numpy as np


class EpsilonGreedyTargetPolicy:
    """
    挙動方策としてEpsilonGreedyを内包した推定方策クラス
    """

    def __init__(self,
                 epsilon: float,
                 decrease_value: float,
                 terminal_value: float,
                 all_state_num: int,
                 all_action_num: int):
        self.q_values = np.array([np.zeros(all_action_num) for _ in range(all_state_num)])
        self.all_state_num = all_state_num
        self.all_action_num = all_action_num
        self.behavior_policy = EpsilonGreedy(epsilon, decrease_value, terminal_value)

    def reset(self) -> None:
        """
        Q値とepsilonの値をresetする
        :return: None
        """
        self.q_values = np.array([np.zeros(self.all_action_num) for _ in range(self.all_state_num)])
        self.behavior_policy.reset()

    def update_behavior_policy(self) -> None:
        """
        挙動方策のパラメータの更新
        :return: None
        """
        self.behavior_policy.update_epsilon()

    def choose(self, state: int) -> int:
        """
        推定方策をpiとしたときのpi(s)に相当するメソッド
        :param state: 状態
        :return: 選択する行動 int
        """
        return self.behavior_policy.choose(self.q_values[state])
