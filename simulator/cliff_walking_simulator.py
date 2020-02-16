from environment.cliff_walking import CliffWalking
from agent.q_learning_agent import QLearningAgent
from agent.sarsa_agent import SarsaAgent
from simulator.base_simulator import BaseSimulator


class CliffWalkingSimulator(BaseSimulator):
    def __init__(self):
        env = CliffWalking()
        agents = list()
        agents.append(QLearningAgent(env.get_all_field_state_num(), env.get_action_num()))
        agents.append(SarsaAgent(env.get_all_field_state_num(), env.get_action_num()))
        super().__init__(env, agents, 100, 500)
