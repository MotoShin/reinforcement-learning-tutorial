from environment.base_environment import BaseEnvironment

import numpy as np

STEP_RESULT = int, float, bool

MAP = ["SHHHHHHHHHHG",
       "FFFFFFFFFFFF",
       "FFFFFFFFFFFF",
       "FFFFFFFFFFFF"]


class CliffWalking(BaseEnvironment):
    """
    sutton本から崖歩きのタスクを参考にした
    """

    def __init__(self):
        super().__init__(MAP)
        self.all_step_num = 0

    def reset(self) -> None:
        self.all_step_num = 0
        super().reset()

    def step(self, a) -> STEP_RESULT:
        next_state, reward, done = super().step(a)
        self.all_step_num += 1

        # 状態数を座標の値へ変換
        row = self.s // self.ncol
        col = self.s % self.ncol

        if b'H' == self.desc[row][col]:
            # 穴の座標だったら報酬が-100
            reward -= 100.0

        if b'G' == self.desc[row][col]:
            # ゴールの座標ならば報酬が12 (本来のタスクのゴール報酬が1なので合計13)
            reward += 12

        # 1ステップ毎に報酬の値を-1する
        return next_state, reward - 1, done

    def get_start_state(self) -> int:
        return np.where(self.isd == 1.0)[0][0]

    def get_all_field_state_num(self) -> int:
        return self.nS

    def get_action_num(self) -> int:
        return self.nA

    def get_all_step_num(self) -> int:
        return self.all_step_num
