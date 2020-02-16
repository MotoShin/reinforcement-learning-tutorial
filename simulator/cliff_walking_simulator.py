from environment.cliff_walking import CliffWalking
from agent.q_learning_agent import QLearningAgent
from agent.sarsa_agent import SarsaAgent
from simulator.base_simulator import BaseSimulator
from util.util import Util

import numpy as np

EXEC_RESULT = dict, dict


class CliffWalkingSimulator(BaseSimulator):
    def __init__(self):
        env = CliffWalking()
        agents = list()
        agents.append(QLearningAgent(env.get_all_field_state_num(), env.get_action_num()))
        agents.append(SarsaAgent(env.get_all_field_state_num(), env.get_action_num()))
        super().__init__(env, agents, 100, 500)

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

                    sum_rewards[episode] += sum_reward / self.env.get_all_step_num()
                    sum_steps[episode] += self.env.get_all_step_num()
                    agent.update_behavior_policy()

            rewards.update({agent.get_agent_name(): sum_rewards / self.all_simulate_number})
            steps.update({agent.get_agent_name(): sum_steps / self.all_simulate_number})
            print()

        return rewards, steps
