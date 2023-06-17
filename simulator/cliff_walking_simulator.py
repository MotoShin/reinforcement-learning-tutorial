from environment.cliff_walking import CliffWalking
from agent.q_learning_agent import QLearningAgent
from agent.soft_q_learning_agent import SoftQLearningAgent
from agent.sarsa_agent import SarsaAgent
from simulator.base_simulator import BaseSimulator
from util.util import Util


class CliffWalkingSimulator(BaseSimulator):
    def __init__(self):
        env = CliffWalking()
        agents = list()
        agents.append(
            QLearningAgent(env.get_all_field_state_num(), env.get_action_num())
        )
        # agents.append(SoftQLearningAgent(env.get_all_field_state_num(), env.get_action_num()))
        agents.append(SarsaAgent(env.get_all_field_state_num(), env.get_action_num()))
        parser = Util.make_config_parser()
        simulation_number = int(parser["BASE"]["SIMULATIONS_NUMBER"])
        episode_number = int(parser["BASE"]["EPISODE"])
        super().__init__(env, agents, simulation_number, episode_number)
