from environment.base_environment import BaseEnvironment

import numpy as np

STEP_RESULT = int, float, bool

MAP = ["SFFF",
       "FFFF",
       "FFFF",
       "FFFG"]


class GridWorld(BaseEnvironment):
    """
    最も簡単なtoyTaskである4*4のグリッドワールド
    """

    def __init__(self):
        super().__init__(MAP)
        self.all_step_num = 0

    def reset(self) -> None:
        self.all_step_num = 0
        super().reset()

    def step(self, a) -> STEP_RESULT:
        # TODO: all_step_numの初期化とか更新を親クラスにまとめたい
        self.all_step_num += 1
        return super().step(a)

    def get_start_state(self) -> int:
        return np.where(self.isd == 1.0)[0][0]

    def get_all_field_state_num(self) -> int:
        return self.nS

    def get_action_num(self) -> int:
        return self.nA

    def get_all_step_num(self) -> int:
        return self.all_step_num
