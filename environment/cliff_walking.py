from environment.base_environment import BaseEnvironment

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
        self.start_state = 0
        self.all_step_num = 0

    def reset(self) -> None:
        self.all_step_num = 0
        super().reset()

    def step(self, a) -> STEP_RESULT:
        next_state, reward, done = super().step(a)
        self.all_step_num += 1
        row = self.s // self.ncol
        col = self.s % self.ncol

        if b'H' == self.desc[row][col]:
            reward -= 100.0

        if b'G' == self.desc[row][col]:
            reward += 12

        return next_state, reward - 1, done

    def get_start_state(self) -> int:
        return self.start_state

    def get_all_field_state_num(self) -> int:
        return self.ncol * self.nrow

    def get_action_num(self) -> int:
        return self.nA

    def get_all_step_num(self) -> int:
        return self.all_step_num
