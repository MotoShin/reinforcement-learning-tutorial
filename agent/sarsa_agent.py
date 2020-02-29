from policy.learning_method.on_policy.sarsa import Sarsa
from agent.base_agent import BaseAgent


class SarsaAgent(BaseAgent):
    """
    SARSAのエージェントのクラス
    """

    def __init__(self, all_state_num: int, all_action_num: int):
        self.learning_method = Sarsa(all_state_num, all_action_num)
        self.start_state = None
        self.current_state = None
        self.current_action = None
        self.memory_action = None

    def reset(self) -> None:
        self.current_state = self.start_state
        self.learning_method.reset()

    def act(self) -> int:
        if self.memory_action is not None:
            self.current_action = self.memory_action
        else:
            self.current_action = self.learning_method.target_policy.choose(self.current_state)
        return self.current_action

    def update_policy(self, reward: int, next_state: int) -> None:
        self.learning_method.update(self.current_state,
                                    self.current_action,
                                    reward,
                                    next_state)
        self.current_state = next_state
        self.memory_action = self.learning_method.target_policy.choose(next_state)

    def update_behavior_policy(self) -> None:
        self.learning_method.update_behavior_policy()
        self.current_state = None
        self.memory_action = None

    def get_agent_name(self) -> str:
        return self.learning_method.get_learning_method_name()

    def set_start_state(self, start_state: int) -> None:
        self.start_state = start_state
