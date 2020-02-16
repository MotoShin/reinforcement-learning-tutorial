from environment.base_environment import BaseEnvironment
from agent.base_agent import BaseAgent
from util.util import Util
from typing import List

import numpy as np

AGENTS = List[BaseAgent]
EXEC_RESULT = dict, dict


class BaseSimulator(object):
    """
    simulatorの処理を記述したクラス
    """

    def __init__(self,
                 env: BaseEnvironment,
                 agents: AGENTS,
                 all_simulate_number: int,
                 all_episode_number: int):
        self.env = env
        self.agents = agents
        self.all_simulate_number = all_simulate_number
        self.all_episode_number = all_episode_number

        for agent in self.agents:
            agent.set_start_state(env.get_start_state())

        parser = Util.make_config_parser()
        self.output_term = int(parser['BASE']['OUTPUT_TERM'])

    def exec(self) -> EXEC_RESULT:
        """
        simulationの内容を記述するメソッド
        :return: list simulationの回数で平均されたepisode毎の報酬が格納されたlist
        """
        sum_rewards = np.zeros(self.all_episode_number)
        sum_steps = np.zeros(self.all_episode_number)

        rewards = {}
        steps = {}
        for agent in self.agents:
            for simulation_number in range(self.all_simulate_number):
                display_simulation_number = simulation_number + 1
                agent.reset()
                # プログラムの進捗表示
                Util.display_progress(self.all_simulate_number,
                                      display_simulation_number,
                                      agent.get_agent_name())

                if display_simulation_number % self.output_term == 0:
                    Util.output_csv({agent.get_agent_name(): sum_rewards},
                                    "process/{0}_{1}.csv".format(agent.get_agent_name(), display_simulation_number))

                for episode in range(self.all_episode_number):
                    sum_reward = 0
                    self.env.reset()

                    while True:
                        chosen_action = agent.act()
                        next_state, reward, done = self.env.step(chosen_action)
                        agent.update_policy(reward, next_state)
                        sum_reward += reward
                        if done:
                            break

                    sum_rewards[episode] += sum_reward
                    sum_steps[episode] += self.env.get_all_step_num()
                    agent.update_behavior_policy()

            rewards.update({agent.get_agent_name(): sum_rewards / self.all_simulate_number})
            steps.update({agent.get_agent_name(): sum_steps / self.all_simulate_number})
            print()

        return rewards, steps
