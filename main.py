from simulator.cliff_walking_simulator import CliffWalkingSimulator
from util.util import Util


if __name__ == '__main__':
    simulator = CliffWalkingSimulator()
    rewards, steps = simulator.exec()

    Util.output_csv(rewards, "result/reward.csv")
    Util.output_csv(steps, "result/steps.csv")

    Util.output_image("reward")
    Util.output_image("steps")
