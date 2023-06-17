import gym
import numpy as np
from abc import ABCMeta, abstractmethod

STEP_RESULT = int, float, bool


class BaseEnvironment(metaclass=ABCMeta):
    """
    Environmentのbaseクラス
    """

    @abstractmethod
    def __init__(self, desc: list):
        self.env = gym.make('FrozenLake-v1', desc=desc, is_slippery=False)
        
        self.desc = desc = np.asarray(desc, dtype="c")
        self.nrow, self.ncol = nrow, ncol = desc.shape

        self.nA = 4
        self.nS = nrow * ncol

        self.isd = np.array(desc == b"S").astype("float64").ravel()
        self.isd /= self.isd.sum()
        pass

    def reset(self) -> None:
        """
        環境の状態をリセットする
        :return: None
        """
        self.s = self.env.reset()[0]

    def step(self, a) -> STEP_RESULT:
        """
        1ステップの処理を記述する
        :param a: int エージェントが行う行動
        :return: s: int 次状態, r: float 報酬, d: bool doneフラグ
        """
        next_state, reward, done, _, _ = self.env.step(a)
        self.s = next_state
        return next_state, reward, done

    @abstractmethod
    def get_start_state(self) -> int:
        """
        初期状態の状態を返す
        :return: int 初期状態のindex
        """
        pass

    @abstractmethod
    def get_all_field_state_num(self) -> int:
        """
        環境の状態の数を返す
        :return: int 環境の状態の数
        """
        pass

    @abstractmethod
    def get_action_num(self) -> int:
        """
        環境の行動数を返す
        :return: int 行動数を返す
        """
        pass

    @abstractmethod
    def get_all_step_num(self) -> int:
        """
        総step数を返す
        :return: int step数を返す
        """
        pass
