import gymnasium as gym
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import random
from gymnasium import Wrapper

T_MAX = int(input("Max number of steps in an episode: "))
NUM_EPISODES = int(input("Number of episodes: "))
GAMMA = float(input("Gamma: "))
LEARNING_RATE = float(input("LEARNING_RATE: "))
EPSILON = float(input("EPSILON: "))
REWARD_THRESHOLD = 100
NUM_EPISODES_TEST = 10000

env = gym.make("Taxi-v3")

def test_episode(agent, env):
    env.reset()
    is_done = False
    t = 0

    while not is_done:
        action = agent.select_action()
        state, reward, is_done, truncated, info = env.step(action)
        t += 1
    return state, reward, is_done, truncated, info

def draw_rewards(rewards):
    data = pd.DataFrame({'Episode': range(1, len(rewards) + 1), 'Reward': rewards})
    plt.figure(figsize=(10, 6))
    sns.lineplot(x='Episode', y='Reward', data=data)

    plt.title('Rewards Over Episodes')
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    plt.grid(True)
    plt.tight_layout()

    plt.show()
    
def print_policy(policy):
    visual_help = {0:'<', 1:'v', 2:'>', 3:'^'}
    policy_arrows = [visual_help[x] for x in policy]
    print(np.array(policy_arrows).reshape([-1, 4]))

class QLearningAgent:
    def __init__(self, env, gamma, learning_rate, epsilon, t_max):
        self.env = env
        self.Q = np.zeros((env.observation_space.n, env.action_space.n))
        self.gamma = gamma
        self.learning_rate = learning_rate
        self.epsilon = epsilon
        self.t_max = t_max
    def change_epsilon(self, epsilon):
        self.epsilon = epsilon
        
    def select_action(self, state, mask,training=True):
        if training and random.random() <= self.epsilon:
            return np.random.choice(self.env.action_space.n)
        else:
            values = np.asarray(self.Q[state,])+np.array(mask)*10
            values[4]=values[4]+0.01*mask[4]
            num=np.argmax(np.random.random(values.shape) * (values==values.max()))
            return num
        
    def update_Q(self, state, action, reward, next_state):
        best_next_action = np.argmax(self.Q[next_state,])
        td_target = reward + self.gamma * self.Q[next_state, best_next_action]
        td_error = td_target - self.Q[state, action]
        self.Q[state, action] += self.learning_rate * td_error
        
    def learn_from_episode(self):
        state, _ = env.reset()
        total_reward = 0
        mask=[1.0,1.0,1.0,1.0,1.0,1.0]
        for i in range(self.t_max):
            action = self.select_action(state,mask)
            new_state, new_reward, is_done, truncated, info = self.env.step(action)

            mask=info['action_mask']

            total_reward += new_reward
            if((state//4)%5!=4):
                estado=state-state%4
                self.update_Q(estado, action, new_reward, new_state)
                self.update_Q(estado+1, action, new_reward, new_state)
                self.update_Q(estado+2, action, new_reward, new_state)
                self.update_Q(estado+3, action, new_reward, new_state)
            else:
                self.update_Q(state, action, new_reward, new_state)
            if is_done:
                break
            state = new_state
        return total_reward

    def policy(self):
        policy = np.zeros(env.observation_space.n) 
        for s in range(env.observation_space.n):
            policy[s] = np.argmax(np.array(self.Q[s]))        
        return policy
    
class CustomTaxiWrapper(Wrapper):
    def __init__(self, env):
        super().__init__(env)

    def step(self, action):
        state, reward, is_done, truncated, info = self.env.step(action)
        if((state//4)%5==4):
            reward=reward+0.9
        return state, reward, is_done, truncated, info
    
fixed_env = CustomTaxiWrapper(env)
agent = QLearningAgent(fixed_env, gamma=GAMMA, learning_rate=LEARNING_RATE, epsilon=EPSILON, t_max=200)
rewards = []
for i in range(NUM_EPISODES):
    reward = agent.learn_from_episode()
    rewards.append(reward)
draw_rewards(rewards)

is_done = False
rewards = []

for n_ep in range(NUM_EPISODES_TEST):
    state, _ = env.reset()
    total_reward = 0
    mask=[1.0,1.0,1.0,1.0,1.0,1.0]
    for i in range(T_MAX):
        action = agent.select_action(state, mask,training=False)
        state, reward, is_done, truncated, info = env.step(action)
        mask=info['action_mask']
        total_reward = total_reward + reward
        env.render()
        if is_done:
            break
    rewards.append(total_reward)
draw_rewards(rewards)
print("Mean Reward")
print(np.mean(rewards))