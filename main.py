from simulator.cliff_walking_simulator import CliffWalkingSimulator
from simulator.grid_world_simulator import GridWorldSimulator
from util.util import Util

import sys


if __name__ == '__main__':
    simulator = None
    if len(sys.argv) == 1:
        print("Usage: python main.py [simulationName]")
        exit()
    else:
        if sys.argv[1] == "gridWorld":
            print("Simulation: Grid World")
            simulator = GridWorldSimulator()
        elif sys.argv[1] == "cliffWalk":
            print("Simulation: Cliff Walk")
            simulator = CliffWalkingSimulator()
        else:
            print("Error simulation name.")
            print("Simulation name is \"gridWorld\" or \"cliffWalk\".")
            exit()

    rewards, steps = simulator.exec()

    Util.output_csv(rewards, "result/reward.csv")
    Util.output_csv(steps, "result/steps.csv")

    Util.output_image("reward")
    Util.output_image("steps")
