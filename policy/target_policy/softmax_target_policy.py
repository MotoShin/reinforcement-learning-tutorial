from policy.behavior_policy.softmax import Softmax

import numpy as np


class SoftmaxTargetPolicy:
    """
    挙動方策としてSoftmaxを内包した推定方策クラス
    """

    def __init__(self,
                 t: float,
                 all_state_num: int,
                 all_action_num: int):
        self.q_values = np.array([np.zeros(all_action_num) for _ in range(all_state_num)])
        self.all_state_num = all_state_num
        self.all_action_num = all_action_num
        self.behavior_policy = Softmax(t)

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
        self.behavior_policy.update()

    def choose(self, state: int) -> int:
        """
        推定方策をpiとしたときのpi(s)に相当するメソッド
        :param state: 状態
        :return: 選択する行動 int
        """
        return self.behavior_policy.choose(self.q_values[state])

    def get_entropy(self, state: int) -> float:
        """
        エントロピーを求めるメソッド
        :param state: 状態
        :return: エントロピー float
        """
        return self.behavior_policy.get_entropy(self.q_values[state])
