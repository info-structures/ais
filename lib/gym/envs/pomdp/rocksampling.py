#Based on the implementation in Julia in the POMDP package
#https://github.com/JuliaPOMDP/RockSample.jl

import numpy as np
from gym import error, spaces, utils
from gym.utils import seeding
import gym
import random
import math

class RockSamplingEnv(gym.Env):
	def __init__(self):
		#config params
		self.map_size = (5, 5)
		self.num_rocks = 3
		rock_positions = [(2,1), (3,3), (1,3)]
		self.rock_positions = [rp[1]+rp[0]*self.map_size[1] for rp in rock_positions]
		#choose to set rocks randomly
		#this random seed allows for changing rock placements randomly
		#it also keeps rock positions same for different env rng
		# rock_sampling_seed = 5
		# random.seed(rock_sampling_seed)
		# self.rock_positions = random.sample(range(self.map_size[0]*self.map_size[1]), self.num_rocks)
		self.init_pos = (0, 0)
		self.good_rock_reward = 20.
		self.bad_rock_reward = -10.
		self.exit_reward = 10.
		self.sensor_efficiency = 20.
		self.discount = 0.95

		self.last_action = -1
		self.start_state = None
		self.current_state = self.start_state
		self.name = "RockSampling"

		self.action_space = spaces.Discrete(5 + self.num_rocks)
		self.state_space = spaces.Discrete(self.map_size[0]*self.map_size[1]*(2**(self.num_rocks)))
		self.observation_space=spaces.Discrete(3)

		self.attempted_rock_sample = False

		#this seed is for env rng
		self.seed()

	def seed(self, seed=None):
		self.np_random, seed = seeding.np_random(seed)
		return [seed]

	def encode_state(self, rover_row, rover_col, rock_status):
		rs_encoding = 0
		for i, rs in enumerate(rock_status):
			rs_encoding = rs_encoding + rs * (2**i)

		rover_encoding = rover_col + rover_row * self.map_size[1]

		state_encoding = int(rs_encoding)*self.map_size[0]*self.map_size[1] + rover_encoding

		return state_encoding

	def decode_state(self, state_encoding):
		rover_encoding = state_encoding % (self.map_size[0]*self.map_size[1])
		rs_encoding = state_encoding // (self.map_size[0]*self.map_size[1])

		rover_col = rover_encoding % self.map_size[1]
		rover_row = rover_encoding // self.map_size[1]

		rock_status = np.zeros(self.num_rocks)
		for i in range(self.num_rocks):
			rock_status[i] = rs_encoding % 2
			rs_encoding = rs_encoding // 2

		return rover_row, rover_col, rock_status

	def get_observation(self, rover_row, rover_col, rock_status, action):
		# obs 0 is good rock
		# obs 1 is bad rock
		# obs 2 is no observation
		rock_to_sense = action - 5
		rock_pos = self.rock_positions[rock_to_sense]
		rock_pos_row = rock_pos // self.map_size[1]
		rock_pos_col = rock_pos % self.map_size[1]

		distance = ((rock_pos_row - rover_row)**2 + (rock_pos_col - rover_col)**2 )**0.5
		efficiency = 0.5 * (1.0 + math.exp(-distance / self.sensor_efficiency))

		if rock_status[rock_to_sense]:
			observation = self.np_random.multinomial(1, [efficiency, 1.-efficiency]).argmax()
		else:
			observation = self.np_random.multinomial(1, [1.-efficiency, efficiency]).argmax()

		return observation

	def step(self, action):
		#0 - north
		#1 - east
		#2 - south
		#3 - west
		#4 - sample
		#5to9 - sense rock data from distance
		done = False
		reward = 0.
		observation = 2 #default is no observation
		rover_row, rover_col, rock_status = self.decode_state(self.current_state)
		rover_encoding = rover_col + rover_row * self.map_size[1]
		
		if action == 0:
			if rover_row != (self.map_size[0]-1):
				rover_row = rover_row + 1
		elif action == 1:
			if rover_col != (self.map_size[1]-1):
				rover_col = rover_col + 1
			else:
				done = True
				reward = self.exit_reward
				observation = -1
		elif action == 2:
			if rover_row != 0:
				rover_row = rover_row - 1
		elif action == 3:
			if rover_col != 0:
				rover_col = rover_col - 1
		elif action == 4:
			if rover_encoding in self.rock_positions:
				sampledrock_index = self.rock_positions.index(rover_encoding)
				if rock_status[sampledrock_index] == 0:
					reward = self.bad_rock_reward
					self.attempted_rock_sample = True
				elif rock_status[sampledrock_index] == 1:
					rock_status[sampledrock_index] = 0
					reward = self.good_rock_reward
					self.attempted_rock_sample = True
		elif (action >=5) and (action < 5+self.num_rocks):
			observation = self.get_observation(rover_row, rover_col, rock_status, action)
		else:
			assert False, 'Invalid action given: `{}`. Action should be from 0-{}'.format(action, 5+self.num_rocks-1)

		next_state = self.encode_state(rover_row, rover_col, rock_status)
		self.last_action = action
		self.current_state = next_state

		return observation, reward, done, {'ars': self.attempted_rock_sample}

	def reset(self):
		self.last_action = -1

		rock_status = self.np_random.randint(2, size = self.num_rocks)
		if np.all(rock_status == 0):
			self.attempted_rock_sample = True
		else:
			self.attempted_rock_sample = False

		rover_row = self.init_pos[0]
		rover_col = self.init_pos[1]

		self.start_state = self.encode_state(rover_row, rover_col, rock_status)
		self.current_state = self.start_state
		observation = 2 #default is no observation

		return observation

	def render(self, mode=None):
		import pygame
		pygame.init()
		from pygame.draw import circle, line

		rover_row, rover_col, rock_status = self.decode_state(self.current_state)

		size = cols, rows = 320, 320
		screen = pygame.display.set_mode(size)
		usable_cols = int(cols * 0.8)
		usable_rows = int(rows * 0.8)
		starting_col = (cols - usable_cols) // 2
		starting_row =  (rows - usable_rows) // 2
		row_steps = usable_rows // (self.map_size[0]-1)
		col_steps = usable_cols // (self.map_size[1]-1)

		white = (255, 255, 255)
		grey = (127, 127, 127)
		red = (255, 0, 0)
		green = (0, 255, 0)
		blue = (0, 0, 255)
		black = (0, 0, 0)

		smalldot_size = 2
		mediumdot_size = 10
		largedot_size = 20

		screen.fill(white)

		circle(screen, blue, (starting_col + rover_col*col_steps, rows - (starting_row + rover_row*row_steps)), largedot_size) #plotting rover
		for i, rp in enumerate(self.rock_positions):
			rs = rock_status[i]
			rock_pos_row = rp // self.map_size[1]
			rock_pos_col = rp % self.map_size[1]
			if rs:
				circle(screen, green, (starting_col + rock_pos_col*col_steps, rows - (starting_row + rock_pos_row*row_steps)), mediumdot_size+2) #good rock
			else:
				circle(screen, red, (starting_col + rock_pos_col*col_steps, rows - (starting_row + rock_pos_row*row_steps)), mediumdot_size+2) #bad rock
			circle(screen, grey, (starting_col + rock_pos_col*col_steps, rows - (starting_row + rock_pos_row*row_steps)), mediumdot_size) #plotting rocks

		for i in range(self.map_size[0]):
			for j in range(self.map_size[1]):
				circle(screen, black, (starting_col + j*col_steps, rows - (starting_row + i*row_steps)), smalldot_size) #plotting grid

		if self.last_action >= 5:
			rp = self.rock_positions[self.last_action-5]
			rock_pos_row = rp // self.map_size[1]
			rock_pos_col = rp % self.map_size[1]
			line(screen, black, (starting_col + rover_col*col_steps, rows - (starting_row + rover_row*row_steps)), (starting_col + rock_pos_col*col_steps, rows - (starting_row + rock_pos_row*row_steps)))

		pygame.display.flip()

#environment tests

#1. encoding decoding test
# sum_stuff = 0
# RS = RockSamplingEnv()
# print (RS.action_space.n)
# print (RS.observation_space.n)
# print (RS.state_space.n)
# obs_space = RS.state_space.n
# for i in range(obs_space):
# 	rover_row, rover_col, rock_status = RS.decode_state(i)
# 	state = RS.encode_state(rover_row, rover_col, rock_status)

# 	print(i, state, rover_row, rover_col, rock_status)
# 	sum_stuff = sum_stuff + abs(i-state)

# print(sum_stuff)


#2. test basic render
# import time

# env = RockSamplingEnv()
# # obs = env.reset()

# for i_episode in range(20):
# 	observation = env.reset()
# 	for t in range(100):
# 		env.render()
# 		time.sleep(2)
# 		action = env.action_space.sample()
# 		observation, reward, done, info = env.step(action)
# 		print('Action: {}, Observation: {}, Reward: {}, Done: {}'.format(action, observation, reward, done))
# 		if done:
# 			print("Episode finished after {} timesteps".format(t+1))
# 			break

# env.close()


#3. limit testing
# import time

# env = RockSamplingEnv()
# # obs = env.reset()

# #try with each action
# # actions = [0]*100

# #go to each rock sample and move out
# # actions = [0, 0, 1, 4, 1, 1, 2, 4, 0, 0, 4, 1, 1, 1, 1, 1]

# #go to each rock, observe and sample (regardless of observation)
# # actions = [0, 0, 1, 5, 4, 1, 1, 0, 6, 4, 2, 2, 7, 4, 1, 1, 1, 1]
# actions = [0, 0, 1, 5, 4, 4, 4, 4, 1, 1, 0, 6, 4, 4, 4, 4, 2, 2, 7, 4, 4, 4, 4, 1, 1, 1, 1]

# for i_episode in range(20):
# 	observation = env.reset()
# 	for t in range(100):
# 		env.render()
# 		time.sleep(2)
# 		action = actions[t]
# 		observation, reward, done, info = env.step(action)
# 		print('Action: {}, Observation: {}, Reward: {}, Done: {}'.format(action, observation, reward, done))
# 		if done:
# 			print("Episode finished after {} timesteps".format(t+1))
# 			break

# env.close()