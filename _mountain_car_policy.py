import gym
import numpy as np

env = gym.make('MountainCar-v0')
s = env.reset()

legal_actions = env.action_space.n
actions = [0, 1, 2]
gamma = 0.99
lr = 0.5
num_episodes = 1000
epsilon = 0.5
epsilon_decay = 0.99
N_BINS = [10, 10]

MIN_VALUES = [0.6, 0.07]
MAX_VALUES = [-1.2, -.07]
BINS = [np.linspace(MIN_VALUES[i], MAX_VALUES[i], N_BINS[i]) for i in range(2)]
rList = []
class QL:
    def __init__(self, Q, policy,
                 legal_actions,
                 actions,
                 gamma,
                 lr):
        self.Q = Q
        self.policy = policy
        self.legal_actions = legal_actions
        self.actions = actions
        self.gamma = gamma
        self.lr = lr

    def q_value(self, s, a):
        if (s, a) in self.Q:
            self.Q[(s, a)]
        else:
            self.Q[s, a] = 0
        return self.Q[s, a]

    def action(self, s):
        if s in self.policy:
            return self.policy[s]
        else:
            self.policy[s] = self.actions[np.random.randint(0, self.legal_actions)]
        return self.policy[s]

    def learn(self, s, a, s1, r, done):
        if done == False:
            self.Q[(s, a)] = self.q_value(s, a) + self.lr * (
                        r + self.gamma * max([self.q_value(s1, a1) for a1 in self.actions]) - self.q_value(s, a))
        else:
            self.Q[(s, a)] = self.q_value(s, a) + self.lr * (r - self.q_value(s, a))
        self.q_values = [self.q_value(s, a1) for a1 in self.actions]
        self.policy[s] = self.actions[self.q_values.index(max(self.q_values))]



def discretize(obs):
    return tuple([int(np.digitize(obs[i], BINS[i])) for i in range(2)])

Q = {}
policy = {}
legal_actions = 3
actions = [0, 1, 2]
gamma = 0.99
lr = 0.5
QL = QL(Q, policy, legal_actions, actions, gamma, lr)
for i in range(num_episodes):
    s_raw = env.reset()
    s = discretize(s_raw)
    rAll = 0
    d = False
    j = 0
    for j in range(200):
        if np.random.random() < epsilon:
            a = np.random.randint(0, legal_actions)
            epsilon = epsilon * epsilon_decay
        else:
            a = QL.action(s)
        s1_raw, r, d, _ = env.step(a)
        rAll = rAll + r
        s1 = discretize(s1_raw)
        env.render()
        if d:
            if rAll < -199:
                r = -100
                QL.learn(s, a, s1, r, d)
                print("Failed! Reward %d" % rAll)
            elif rAll > -199:
                print("Passed! Reward %d" % rAll)
            break
        QL.learn(s, a, s1, r, d)
        if j == 199:
            print("Reward %d after full episode" % (rAll))

        s = s1
env.close()
