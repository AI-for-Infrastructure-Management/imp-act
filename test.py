from environment import RoadEnvironment

def main():
    env = RoadEnvironment()
    env.reset()
    actions = [[1,1] for _ in range(4)]
    done = False
    while not done:
        observation, cost, done, info = env.step(actions)
        # print(observation, cost, done, info)
        print(observation["edge_observations"], cost, done, info["all_beliefs"])


if __name__ == "__main__":
    main()