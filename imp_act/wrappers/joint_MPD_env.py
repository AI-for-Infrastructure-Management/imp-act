import itertools


class JointMDPEnv:
    def __init__(self, env):
        self.base_env = env
        self.num_components = sum(env.edge_segments_numbers)
        self.time_horizon = env.max_timesteps + 1  # +1 for terminal state
        component_actions = [0, 2, 3]
        num_component_actions = len(component_actions)
        num_damage_states = 4

        # cartesian product of damage states
        _joint = itertools.product(range(num_damage_states), repeat=self.num_components)

        # cartesian product of _joint and time horizon
        self.joint_state_space = list(
            itertools.product(range(self.time_horizon), _joint, repeat=1)
        )

        # cartesian product of actions
        self.joint_action_space = list(
            itertools.product(component_actions, repeat=self.num_components)
        )

        # compute size of joint state and action spaces
        self.num_joint_states = (
            num_damage_states**self.num_components * self.time_horizon
        )
        self.num_joint_actions = num_component_actions**self.num_components

        # print some joint state and action space info
        print(f"num_joint_states: {self.num_joint_states}")
        print(f"num_joint_actions: {self.num_joint_actions}")

        self.reset()

    def encode_state(self, timestep, state):
        _state = (timestep, tuple([item for sub_list in state for item in sub_list]))
        return self.joint_state_space.index(_state)

    def decode_state(self, state_idx):
        return self.joint_state_space[state_idx]

    def encode_action(self, action):
        return self.joint_action_space.index(action)

    def decode_action(self, action_idx):
        action = self.joint_action_space[action_idx]
        return [[a] for a in action]

    def reset(self):
        self.base_env.reset()
        self.timestep = 0
        self.state = self.base_env._get_states()
        return self.encode_state(self.timestep, self.state)

    def step(self, action_idx):
        self.timestep += 1
        _, reward, done, info = self.base_env.step(self.decode_action(action_idx))
        self.state = self.base_env._get_states()
        return self.encode_state(self.timestep, self.state), reward, done, info
