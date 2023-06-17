from simulator.grid_world_simulator import GridWorldSimulator
from util.util import Util


if __name__ == '__main__':
    simulator = GridWorldSimulator()

    rewards, steps, entropy = simulator.exec()

    Util.output_csv(rewards, "result/reward.csv")
    Util.output_csv(steps, "result/steps.csv")
    Util.output_csv(entropy, "result/entropy.csv")

    Util.output_image("reward")
    Util.output_image("steps")
    Util.output_image("entropy")
