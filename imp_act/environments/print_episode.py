import numpy as np

from imp_act import make


def main():
    env = make("ToyNetwork-v1")
    env.reset()
    # actions = [[1, 1] for _ in range(4)]
    actions = [[0] for _ in range(6)]
    done = False
    timestep = 0
    while not done:
        timestep += 1
        observation, reward, done, info = env.step(actions)
        # print(observation, reward, done, info)
        print("=" * 50)
        print(f"timestep: {timestep}")
        print(f"reward: {reward}")
        print(f"done: {done}")
        print(f"total travel time: {info['total_travel_time']}")
        print(f'rewards: {info["reward_elements"]}')

        for i, observations, beliefs, states, volumes, travel_times in zip(
            range(len(observation["edge_observations"])),
            observation["edge_observations"],
            observation["edge_beliefs"],
            info["states"],
            info["volumes"],
            info["travel_times"]
        ):

            print(f"\nedge: {i}")
            print(f"states:       {states}")
            print(f"observations: {observations}")
            print(f"beliefs:      {[list(np.round(belief,2)) for belief in beliefs]}")
            print(f"volumes:      {volumes}")
            print(f"travel times: {travel_times}")

if __name__ == "__main__":
    main()
