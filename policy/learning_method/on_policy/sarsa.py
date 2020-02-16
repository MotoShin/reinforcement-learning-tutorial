from policy.target_policy.epsilon_greedy_target_policy import EpsilonGreedyTargetPolicy
from policy.learning_method.base_learning_method import BaseLearningMethod
from util.util import Util


class Sarsa(BaseLearningMethod):
    """
    on_policyのTD学習であるSarsaのQ値の更新を記述するクラス
    """

    def __init__(self, all_state_num: int, all_action_num: int):
        parser = Util.make_config_parser()
        self.target_policy = EpsilonGreedyTargetPolicy(
            float(parser['BASE']['EPSILON']),
            float(parser['BASE']['EPSILON_DECREASE_VALUE']),
            float(parser['BASE']['EPSILON_TERMINAL_VALUE']),
            all_state_num,
            all_action_num
        )
        self.learning_method_name = parser['BASE']['SARSA']
        self.ALPHA = float(parser['BASE']['ALPHA'])
        self.GAMMA = float(parser['BASE']['GAMMA'])

    def reset(self) -> None:
        self.target_policy.reset()

    def update(self,
               current_state: int,
               action: int,
               reward: float,
               next_state: int) -> None:
        # 自身の挙動方策に従って次状態の行動を選択する
        next_action = self.target_policy.choose(next_state)
        # TD誤差の導出
        td_error = \
            reward + self.GAMMA * \
            self.target_policy.q_values[next_state, next_action] \
            - self.target_policy.q_values[current_state, action]
        # Q値の更新
        self.target_policy.q_values[current_state, action] += self.ALPHA * td_error

    def update_behavior_policy(self) -> None:
        self.target_policy.update_behavior_policy()

    def get_learning_method_name(self) -> str:
        return self.learning_method_name
