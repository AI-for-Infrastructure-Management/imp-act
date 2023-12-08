from environment import RoadEnvironment
import numpy as np

def main():
    env = RoadEnvironment()
    env.reset()
    actions = [[1,1] for _ in range(4)]
    done = False
    timestep = 0
    while not done:
        timestep += 1
        observation, cost, done, info = env.step(actions)
        # print(observation, cost, done, info)
        print("="*50)
        print(f"timestep: {timestep}")
        print(f"cost: {cost}")
        print(f"done: {done}")

        for i, observations, beliefs, states in zip(
                range(len(observation["edge_observations"])),
                observation["edge_observations"], 
                observation["edge_beliefs"], 
                info["states"]):
            
            print(f"\nedge: {i}")
            print(f"states:       {states}")
            print(f"observations: {observations}")
            print(f"beliefs:      {[list(np.round(belief,2)) for belief in beliefs]}")
            print(f"total travel time: {info['total_travel_time']}")


if __name__ == "__main__":
    main()