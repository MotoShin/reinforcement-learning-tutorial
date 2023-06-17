import numpy as np
import random
from policy.behavior_policy.base_behavior_policy import BaseBehaviorPolicy


class EpsilonGreedy(BaseBehaviorPolicy):
    """
    EpsilonGreedyの処理内容が書かれているクラス
    epsilon, decrease_value, terminal_valueを全て0にするとgreedyになる
    """

    def __init__(self, epsilon: float, decrease_value: float, terminal_value: float):
        self.epsilon = epsilon
        self.first_epsilon = epsilon
        self.decrease_value = decrease_value
        self.terminal_value = terminal_value

    def reset(self) -> None:
        """
        Epsilonの値をリセットする
        :return: None
        """
        self.epsilon = self.first_epsilon

    def update(self) -> None:
        """
        Epsilonの値を更新する
        :return: None
        """
        self.epsilon = max(self.epsilon - self.decrease_value, self.terminal_value)

    def choose(self, ary) -> int:
        """
        epsilonの確率でランダムに選択し(1-epsilon)の確率でgreedyに選択する
        :param ary: ndarray
        :return: 選択した配列のindexの値 int
        """
        if random.random() < self.epsilon:
            selected = np.random.choice(np.arange(len(ary)))
        else:
            idx = np.where(ary == max(ary))
            selected = np.random.choice(idx[0])

        return selected
