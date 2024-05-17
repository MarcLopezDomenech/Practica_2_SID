import gymnasium as gym
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import collections
from time import time

class DirectEstimationAgent:
    def __init__(self, env, gamma, num_trajectories):
        self.env = env
        self.state, _ = self.env.reset()
        self.rewards = collections.defaultdict(float)
        self.transits = collections.defaultdict(collections.Counter)
        self.V = np.zeros(self.env.observation_space.n)
        self.gamma = gamma
        self.num_trajectories = num_trajectories

    def play_n_random_steps(self, count):
        for _ in range(count):
            action = self.env.action_space.sample()
            new_state, reward, is_done, truncated, _ = self.env.step(action)
            self.rewards[(self.state, action, new_state)] = reward
            self.transits[(self.state, action)][new_state] += 1
            if is_done:
                self.state, _ = self.env.reset() 
            else: 
                self.state = new_state

    def calc_action_value(self, state, action):
        target_counts = self.transits[(state, action)]
        total = sum(target_counts.values())
        action_value = 0.0
        for s_, count in target_counts.items():
            r = self.rewards[(state, action, s_)]
            prob = (count / total)
            action_value += prob*(r + self.gamma * self.V[s_])
        return action_value

    def select_action(self, state):
        best_action, best_value = None, None
        for action in range(self.env.action_space.n):
            action_value = self.calc_action_value(state, action)
            if best_value is None or best_value < action_value:
                best_value = action_value
                best_action = action
        return best_action

    def value_iteration(self):
        self.play_n_random_steps(self.num_trajectories)
        max_diff = 0
        for state in range(self.env.observation_space.n):
            state_values = [
                self.calc_action_value(state, action)
                for action in range(self.env.action_space.n)
            ]
            new_V = max(state_values)
            diff = abs(new_V - self.V[state])
            if diff > max_diff:
                max_diff = diff
            self.V[state] = new_V
        return self.V, max_diff
    
    def policy(self):   
        policy = np.zeros(env.observation_space.n) 
        for s in range(env.observation_space.n):
            Q_values = [self.calc_action_value(s,a) for a in range(self.env.action_space.n)] 
            policy[s] = np.argmax(np.array(Q_values))        
        return policy

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
    
def check_improvements(agent):
    reward_test = 0.0
    time_test = 0.0
    for i in range(NUM_EPISODES):
        total_reward = 0.0
        start_time = time()
        state, _ = env.reset()
        for i in range(T_MAX):
            action = agent.select_action(state)
            new_state, new_reward, is_done, truncated, _ = env.step(action)
            total_reward += new_reward
            if is_done: 
                break
            state = new_state
        reward_test += total_reward
        time_test += time() - start_time
    reward_avg = reward_test / NUM_EPISODES
    return reward_avg, time_test

def train(agent): 
    rewards = []
    max_diffs = []
    t = 0
    best_reward = 0.0
    time_total = 0.0
    times_episode_avg = []
     
    while best_reward < REWARD_THRESHOLD:
        _, max_diff = agent.value_iteration()
        max_diffs.append(max_diff)
        print("After value iteration, max_diff = " + str(max_diff))
        t += 1
        reward_test, time_test = check_improvements(agent)
        rewards.append(reward_test)
               
        if reward_test > best_reward:
            #print(f"Best reward updated {reward_test:.2f} at iteration {t}") 
            best_reward = reward_test
        time_total  += time_test
        times_episode_avg.append(time_test / NUM_EPISODES)
    
    return time_total, sum(times_episode_avg) / len(times_episode_avg), rewards, max_diffs

def print_policy(policy):
    visual_help = {0:'<', 1:'v', 2:'>', 3:'^', 4:'P', 5:'L'}
    policy_arrows = [visual_help[x] for x in policy]
    print(np.array(policy_arrows).reshape([-1, 4]))

T_MAX = int(input("Max number of steps in an episode: "))
NUM_EPISODES = int(input("Number of episodes: "))
GAMMA = float(input("Gamma: "))
REWARD_THRESHOLD = float(input("Reward threshold: "))
N = int(input("Number of experiments: "))

env = gym.make("Taxi-v3")
rewards_exp = []
times_total_exp = []
times_episodes_avg = []

agent = DirectEstimationAgent(env, gamma=GAMMA, num_trajectories = NUM_EPISODES)
times_total, time_avg, rewards, max_diff = train(agent)
draw_rewards(rewards)

rewards = []
for n_ep in range(N):
    state, _ = env.reset()
    print('Episode: ', n_ep)
    total_reward = 0
    for i in range(T_MAX):
        action = agent.select_action(state)
        state, reward, is_done, truncated, _ = env.step(action)
        total_reward = total_reward + reward
        if is_done:
            break
    rewards.append(total_reward)

draw_rewards(rewards)

filename = str(T_MAX)+'_'+str(NUM_EPISODES)+'_'+str(GAMMA)+'_'+str(REWARD_THRESHOLD)+'_'+str(N)
with open("./results_taxi-v3_direct-estimation_"+filename+".txt", "w") as f:
	f.write("T_MAX: " + str(T_MAX) + "\n")
	f.write("NUM_EPISODES: " + str(NUM_EPISODES) + "\n")
	f.write("GAMMA: " + str(GAMMA) + "\n")
	f.write("REWARD_THRESHOLD: " + str(REWARD_THRESHOLD) + "\n")
	f.write("N: " + str(N) + "\n")
	f.write("\n\n")
	f.write("Columns: Iteration | Reward\n")
	for i in range(N):
		f.write(str(i) + '\t' + str(rewards[i]) +"\n")
