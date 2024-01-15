import numpy as np
import random


class Q_learner:
    def __init__(
        self,
        env,
        num_episodes=1000,
        discount_factor=0.9,
        lr_start=1,
        lr_end=0.3,
        epsilon_start=0.1,
        epsilon_end=0.01,
    ):
        self.env = env
        self.discount_factor = discount_factor
        self.lr = lr_start
        self.lr_end = lr_end
        self.lr_decay = (lr_start - lr_end) / num_episodes
        self.epsilon = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = (epsilon_start - epsilon_end) / num_episodes
        self.num_episodes = num_episodes
        self.Q = np.ones((env.state_space.n, env.action_space.n))

    def greedy_policy(self, state):
        q_values = self.Q[state, :]
        greedy_actions = np.flatnonzero(q_values == max(q_values))

        return np.random.choice(greedy_actions)

    def epsilon_greedy_policy(self, state, epsilon):
        if random.uniform(0, 1) < epsilon:
            return self.env.action_space.sample()
        else:
            return self.greedy_policy(state)

    def _train_episode(self, verbose=False):
        state = self.env.reset()
        done = False
        total_reward = 0

        while not done:
            action = self.epsilon_greedy_policy(state, self.epsilon)
            next_state, reward, done, _ = self.env.step(action)

            ####### Q-learning update #######

            # compute value of next_state
            # V(s') = max_a' Q(s', a')
            future_value = np.max(self.Q[next_state, :]) * (1 - done)

            # TD target = r + γ max_a' Q(s', a')
            TD_target = reward + self.discount_factor * future_value

            # update Q table
            # Q(s,a) = Q(s, a) + α (TD_target - Q(s,a))
            self.Q[state, action] += self.lr * (TD_target - self.Q[state, action])

            # update state
            state = next_state

            # update epsilon
            self.epsilon = max(self.epsilon_end, self.epsilon - self.epsilon_decay)

            # update learning rate
            self.lr = max(self.lr_end, self.lr - self.lr_decay)
            
            # update total reward
            total_reward += reward

        if verbose:
            print(f'Episode {self.episode} | Total reward: {total_reward}')

    def train(self, verbose=False):
        for episode in range(self.num_episodes):
            self.episode = episode
            self._train_episode(verbose)

        return self.Q, self.greedy_policy
