from policy.target_policy.softmax_target_policy import SoftmaxTargetPolicy
from policy.learning_method.base_learning_method import BaseLearningMethod
from util.util import Util

import numpy as np


class SoftQLearning(BaseLearningMethod):
    """
    off-policyのTD学習であるQ学習のQ値の更新方法を記述するクラス
    """

    def __init__(self, all_state_num: int, all_action_num: int):
        # TODO: 設定値はコマンドライン引数で指定できるようにする
        parser = Util.make_config_parser()
        self.target_policy = SoftmaxTargetPolicy(
            float(parser['BASE']['SOFT_T']),
            all_state_num,
            all_action_num
        )
        self.learning_method_name = parser['BASE']['SOFT_Q_LERNING']
        self.ALPHA = float(parser['BASE']['ALPHA'])
        self.GAMMA = float(parser['BASE']['GAMMA'])

    def reset(self) -> None:
        self.target_policy.reset()

    def update(self,
               current_state: int,
               action: int,
               reward: float,
               next_state: int) -> None:
        # 次状態の行動の中でq値が最大のものを取得
        q_values = self.target_policy.q_values[next_state]
        next_action = np.random.choice(np.where(q_values == max(q_values))[0])
        # TD誤差の導出
        td_error = \
            reward + self.GAMMA * \
            self.target_policy.q_values[next_state, next_action] - self.target_policy.q_values[current_state, action]
        # Q値の更新
        self.target_policy.q_values[current_state, action] += self.ALPHA * td_error

    def update_behavior_policy(self) -> None:
        self.target_policy.update_behavior_policy()

    def get_learning_method_name(self) -> str:
        return self.learning_method_name
