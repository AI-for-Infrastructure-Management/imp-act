import numpy as np

from imp_act import make


def main():
    env = make("ToyExample-v2")
    print(env.get_count_redundancies_summary())
    print(env.get_edge_traffic_summary())
    env.reset()
    actions = [[1] * len(e["road_edge"].segments) for e in env.graph.es]
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
        print(
            f"remaining budget: {observation['budget_remaining']}/{env.budget_amount}"
        )
        print(f"time until budget renewal: {observation['budget_time_until_renewal']}")

        for i, observations, beliefs, states in zip(
            range(len(observation["edge_observations"])),
            observation["edge_observations"],
            observation["edge_beliefs"],
            info["states"],
        ):

            print(f"\nedge: {i}")
            print(f"states:       {states}")
            print(f"observations: {observations}")
            print(f"beliefs:      {[list(np.round(belief,2)) for belief in beliefs]}")


if __name__ == "__main__":
    main()