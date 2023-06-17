from policy.behavior_policy.epsilon_greedy import EpsilonGreedy

import numpy as np
from scipy.stats import entropy


class EpsilonGreedyTargetPolicy:
    """
    挙動方策としてEpsilonGreedyを内包した推定方策クラス
    """

    def __init__(
        self,
        epsilon: float,
        decrease_value: float,
        terminal_value: float,
        all_state_num: int,
        all_action_num: int,
    ):
        self.q_values = np.array(
            [np.zeros(all_action_num) for _ in range(all_state_num)]
        )
        self.all_state_num = all_state_num
        self.all_action_num = all_action_num
        self.behavior_policy = EpsilonGreedy(epsilon, decrease_value, terminal_value)

    def reset(self) -> None:
        """
        Q値とepsilonの値をresetする
        :return: None
        """
        self.q_values = np.array(
            [np.zeros(self.all_action_num) for _ in range(self.all_state_num)]
        )
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

    def get_entropy(self, state: int) -> int:
        """
        方策のエントロピーを算出する
        :param ary: ndarray
        :return: 選択した配列のindexの値 int
        """
        probs = self._softmax(self.q_values[state])
        if np.sum(probs) == 0:
            return 0
        else:
            return entropy(probs, base=len(self.q_values[state]))

    def _softmax(self, ary: np.ndarray) -> list:
        # TODO: まとめたい
        """
        Sotfmax関数
        :param ary: ndarray
        :return: 入力された配列を確率に変換したもの
        """
        max_num = max(ary)
        x = np.exp((ary - max_num) / 1.0)
        u = np.sum(x)
        return x / u
