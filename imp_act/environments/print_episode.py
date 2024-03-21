from imp_act import make


def do_nothing_policy(observation):
    return [[0] * len(e) for e in observation["edge_observations"]]


def fail_replace_policy(observation):
    actions = []
    for edge in observation["edge_observations"]:
        edge_action = []
        for segment in edge:
            if segment >= 3:
                edge_action.append(3)
            else:
                edge_action.append(0)
        actions.append(edge_action)
    return actions


def heuristic_policy(observation):
    actions = []
    current_time = observation["time_step"]
    for edge in observation["edge_observations"]:
        edge_actions = []
        for segment in edge:
            if segment >= 5:
                edge_actions.append(4)  # Reconstruction
            elif segment >= 5:
                edge_actions.append(3)  # Major repair
            elif segment >= 1:
                edge_actions.append(2)  # Minor repair
            elif current_time % 1 == 0:
                edge_actions.append(1)  # Inspection
            else:
                edge_actions.append(0)  # Do nothing
        actions.append(edge_actions)
    return actions


def main():
    env = make("Denmark-v1")
    obs = env.reset()
    done = False
    timestep = 0
    total_reward = 0
    policy = heuristic_policy
    while not done:
        timestep += 1
        actions = policy(obs)
        obs, reward, done, info = env.step(actions)
        total_reward += reward

        print(f"timestep: {timestep}")
        print(f"reward: {reward:.2e}")
        print(f"travel_time_reward: {info['reward_elements'][0]:.2e}")
        print(f"maintenance_reward: {info['reward_elements'][1]:.2e}")
        print(f"total travel time: {info['total_travel_time']}")

        """for i, observations, beliefs, states in zip(
            range(len(observation["edge_observations"])),
            observation["edge_observations"],
            observation["edge_beliefs"],
            info["states"],
        ):

            print(f"\nedge: {i}")
            print(f"states:       {states}")
            print(f"observations: {observations}")
            print(f"beliefs:      {[list(np.round(belief,2)) for belief in beliefs]}")"""

        print("=" * 50)

    print(f"total reward: {total_reward:.3e}")


if __name__ == "__main__":
    main()
