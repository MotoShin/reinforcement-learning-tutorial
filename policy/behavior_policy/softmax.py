import numpy as np
import random
from scipy.stats import entropy
from policy.behavior_policy.base_behavior_policy import BaseBehaviorPolicy


class Softmax(BaseBehaviorPolicy):
    """
    EpsilonGreedyの処理内容が書かれているクラス
    epsilon, decrease_value, terminal_valueを全て0にするとgreedyになる
    """

    def __init__(self, t: float):
        self.soft_t = t

    def reset(self) -> None:
        pass

    def update(self) -> None:
        pass

    def choose(self, ary) -> int:
        """
        Q値からSoftmax関数を使用して確率値を求めて確率的に行動選択を行う
        :param ary: ndarray
        :return: 選択した配列のindexの値 int
        """
        probs = self._softmax(ary)
        return random.choices(np.arange(len(ary)), weights=probs)[0]

    def get_entropy(self, ary) -> float:
        """
        方策のエントロピーを算出する
        :param ary: ndarray
        :return: 選択した配列のindexの値 int
        """
        if np.sum(ary) == 0:
            return 0
        else:
            return entropy(np.array(ary), base=len(ary))

    def _softmax(self, ary: np.ndarray) -> list:
        """
        Sotfmax関数
        :param ary: ndarray
        :return: 入力された配列を確率に変換したもの
        """
        max_num = max(ary)
        x = np.exp((ary - max_num) / self.soft_t)
        u = np.sum(x)
        return x / u
