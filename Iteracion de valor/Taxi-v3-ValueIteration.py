import gymnasium as gym
import numpy as np
from time import time

class ValueIterationAgent:
	def __init__(self, env, gamma):
		self.env = env
		self.V = np.zeros(self.env.observation_space.n)
		self.gamma = gamma
		
	def calc_action_value(self, state, action):
		action_value = sum([prob * (reward + self.gamma * self.V[next_state])
							for prob, next_state, reward, _ 
							in self.env.unwrapped.P[state][action]]) 
		return action_value

	def select_action(self, state):
		best_action = best_value = None
		for action in range(self.env.action_space.n):
			action_value = self.calc_action_value(state, action)
			if not best_value or best_value < action_value:
				best_value = action_value
				best_action = action
		return best_action

	def value_iteration(self):
		for state in range(self.env.observation_space.n):
			state_values = []
			for action in range(self.env.action_space.n):  
				state_values.append(self.calc_action_value(state, action))
			new_V = max(state_values)
			self.V[state] = new_V
	
	def policy(self):   
		policy = np.zeros(env.observation_space.n)
		for s in range(env.observation_space.n):
			Q_values = [self.calc_action_value(s,a) for a in range(self.env.action_space.n)] 
			policy[s] = np.argmax(np.array(Q_values))		
		return policy
	
def check_improvements():
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
	t = 0
	best_reward = 0.0
	time_total = 0.0
	times_episode_avg = []

	while best_reward < REWARD_THRESHOLD:
		agent.value_iteration()
		t += 1
		reward_test, time_test = check_improvements()
			   
		if reward_test > best_reward:
			best_reward = reward_test
		
		# print("reward threshold:", REWARD_THRESHOLD, "best reward", best_reward, "reward test", reward_test)

		time_total += time_test
		times_episode_avg.append(time_test / NUM_EPISODES)
	
	return time_total, sum(times_episode_avg) / len(times_episode_avg)

def print_policy(policy):
	visual_help = {0:'v', 1:'^', 2:'>', 3:'<', 4:'p', 5:'d'}
	policy_arrows = [visual_help[x] for x in policy]
	print(np.array(policy_arrows).reshape([-1, 4]))
	
def test_policy(env, agent):
	is_done = False
	rewards = []
	for n_ep in range(NUM_EPISODES):
		state, _ = env.reset()
		total_reward = 0
		for i in range(T_MAX):
			action = agent.select_action(state)
			state, reward, is_done, truncated, _ = env.step(action)
			total_reward = total_reward + reward
			if is_done:
				break
		rewards.append(total_reward)
	return rewards


T_MAX = int(input("Max number of steps in an episode: ")) # 15
NUM_EPISODES = int(input("Number of episodes: ")) # 5
GAMMA = float(input("Gamma: ")) # 0.95
REWARD_THRESHOLD = float(input("Reward Threshold: ")) # 0.9
N = int(input("Number of experiments: "))

env = gym.make('Taxi-v3')

rewards_exp = []
times_total_exp = []
times_episodes_avg_exp = []
for i in range(N):
	print("Iteration:", i)
	agent = ValueIterationAgent(env, gamma=GAMMA)
	times_total, times_episode_avg = train(agent)
	rewards_test = test_policy(env, agent)

	times_total_exp.append(times_total)
	times_episodes_avg_exp.append(times_episode_avg)
	rewards_exp.append(rewards_test)

filename = str(T_MAX)+'_'+str(NUM_EPISODES)+'_'+str(GAMMA)+'_'+str(REWARD_THRESHOLD)+'_'+str(N)
with open("./results_taxi-v3_value-iteration_"+filename+".txt", "w") as f:
	f.write("T_MAX: " + str(T_MAX) + "\n")
	f.write("NUM_EPISODES: " + str(NUM_EPISODES) + "\n")
	f.write("GAMMA: " + str(GAMMA) + "\n")
	f.write("REWARD_THRESHOLD: " + str(REWARD_THRESHOLD) + "\n")
	f.write("N: " + str(N) + "\n")
	f.write("\n\n")
	f.write("Columns: Iteration | Reward | Total time | Avg time episode\n")
	for i in range(N):
		f.write(str(i) + '\t' + str(rewards_exp[i]) + '\t' + str(times_total_exp[i]) + '\t' + str(times_episodes_avg_exp[i]) + "\n")

# policy = agent.policy()
# print_policy(policy)