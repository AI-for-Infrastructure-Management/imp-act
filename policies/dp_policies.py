import numpy as np
from .agent import Agent
from environment import RoadSegment

class OptimalMDPAgent(Agent):
    """Agent that observes the hidden states and uses the optimal MDP policy based on the tranistion tables"""
    def __init__(self, horizon="finite", max_timestep=50, seed=42, iterations_Q=5000, gamma=0.95):
        rng = np.random.default_rng(seed)
        segment = RoadSegment(rng)
        transition_tables = segment.transition_tables
        state_action_rewards = segment.state_action_reward
        state_action_rewards[-1, :-1] = np.array([-1000]*3)
        if horizon=="finite":
            self.Q = self.finite_horizon_policy(transition_tables, state_action_rewards, max_timestep)
        elif horizon=="infinite":
            self.Q = self.infinite_horizon_policy(transition_tables, state_action_rewards, iterations_Q, gamma)
        self.horizon = horizon
        self.timestep = 0

    def bellmanOperator(self, transition_tables, state_action_rewards, V):
        S, A = state_action_rewards.shape 
        Q = np.empty((A, S))
        for a in range(A):
            Q[a] = state_action_rewards[:, a] + transition_tables[a].dot(V) 
            #print(Q)
        return Q, Q.max(axis=0)

    def finite_horizon_policy(self, transition_tables, state_action_rewards, N):
        S, A = state_action_rewards.shape
        Q = np.empty((A, S , N))
        V = np.zeros((S, N+1))
        for i in range(N):
            q, v = self.bellmanOperator(transition_tables, state_action_rewards, V[:, N - i])
            stage = N - i - 1
            V[:, stage] = v
            Q[:, :, stage] = q
        return Q
    
    def infinite_horizon_policy(self, transition_tables, state_action_rewards, iterations, gamma):
        S, A = state_action_rewards.shape
        Q = np.zeros((A, S))
        for _ in range(iterations):
            Q_prev = Q
            for s in range(S):
                for a in range(A):
                    q = np.sum(transition_tables[a, s, s_next] 
                               * (state_action_rewards[s, a] + gamma * Q_prev[:, s_next].max()) for s_next in range(S))
                    Q[a, s] = q
        return Q
    
    def get_action(self, state):
        if self.horizon=="finite":
            Q = self.Q[:, :, self.timestep]
        else:
            Q = self.Q
        policy = Q.argmax(0)
        actions = policy[np.array(state)]
        self.timestep += 1
        return actions

    def reset(self, observation):
        self.timestep = 0

