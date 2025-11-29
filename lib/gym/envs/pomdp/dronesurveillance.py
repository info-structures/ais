#Based on the implementation in Julia in the POMDP package
#https://github.com/JuliaPOMDP/DroneSurveillance.jl

import numpy as np
import gymnasium as gym
from gymnasium import error, spaces, utils
from gymnasium.utils import seeding

# drone/ground_bot row, col positions are measured from bottom left (0, 0)
#ground_bot cannot transition to (0, 0) or (4, 4) - (goal locations)
#ground_bot spawns anywhere except (0, 0) or (4, 4)
#or (1, 0), (0, 1), (1, 1) - (starting FOV regions)

class DroneSurveillanceEnv(gym.Env):
	def __init__(self):
		#config params
		self.map_size = (5, 5)
		self.init_pos = (0, 0)
		self.discount = 0.95

		self.bad_exit_reward = 0.
		self.position_match_reward = -1.
		self.completion_reward = 1.

		self.start_state = None
		self.current_state = self.start_state
		self.name = "DroneSurveillance"
		
		self.action_space = spaces.Discrete(5)
		self.state_space = spaces.Discrete(self.map_size[0]*self.map_size[1]*(self.map_size[0]*self.map_size[1]-2))
		self.observation_space = spaces.Discrete(10)

	def encode_state(self, drone_row, drone_col, ground_bot_row, ground_bot_col):
		d_encoding = drone_col + drone_row*self.map_size[1]

		if ground_bot_row == 0:
			g_encoding = ground_bot_col-1
		else:
			g_encoding = (self.map_size[1]-1) + (ground_bot_row-1)*self.map_size[1] + ground_bot_col

		state_encoding = g_encoding*self.map_size[0]*self.map_size[1] + d_encoding
		return state_encoding

	def decode_state(self, state_encoding):
		d_encoding = state_encoding % (self.map_size[0]*self.map_size[1])
		g_encoding = state_encoding // (self.map_size[0]*self.map_size[1])
		drone_col = d_encoding % self.map_size[1]
		drone_row = d_encoding // self.map_size[1]
		ground_bot_col = (g_encoding+1) % self.map_size[1]
		ground_bot_row = (g_encoding+1) // self.map_size[1]

		return drone_row, drone_col, ground_bot_row, ground_bot_col

	def get_observation(self, drone_row, drone_col, ground_bot_row, ground_bot_col):
		# 0 is not visible
		# 7   8    9
		# 4  *(5)  6
		# 1   2    3
		observation = 0
		if (drone_row-1 == ground_bot_row) and (drone_col-1 == ground_bot_col):
			observation = 1
		elif (drone_row-1 == ground_bot_row) and (drone_col == ground_bot_col):
			observation = 2
		elif (drone_row-1 == ground_bot_row) and (drone_col+1 == ground_bot_col):
			observation = 3
		elif (drone_row == ground_bot_row) and (drone_col-1 == ground_bot_col):
			observation = 4
		elif (drone_row == ground_bot_row) and (drone_col == ground_bot_col):
			observation = 5
		elif (drone_row == ground_bot_row) and (drone_col+1 == ground_bot_col):
			observation = 6
		elif (drone_row+1 == ground_bot_row) and (drone_col-1 == ground_bot_col):
			observation = 7
		elif (drone_row+1 == ground_bot_row) and (drone_col == ground_bot_col):
			observation = 8
		elif (drone_row+1 == ground_bot_row) and (drone_col+1 == ground_bot_col):
			observation = 9

		return observation

	def step(self, action):
		#0 - north
		#1 - east
		#2 - south
		#3 - west
		#4 - hover
		done = False
		drone_row, drone_col, ground_bot_row, ground_bot_col = self.decode_state(self.current_state)
		if action == 0:
			if drone_row != (self.map_size[0]-1):
				drone_row = drone_row + 1
			else:
				done = True
		elif action == 1:
			if drone_col != (self.map_size[1]-1):
				drone_col = drone_col + 1
			else:
				done = True
		elif action == 2:
			if drone_row != 0:
				drone_row = drone_row - 1
			else:
				done = True
		elif action == 3:
			if drone_col != 0:
				drone_col = drone_col - 1
			else:
				done = True
		elif action == 4:
			drone_row = drone_row
			drone_col = drone_col
		else:
			assert False, 'Invalid action given: `{}`. Action should be from 0-4'.format(action)

		if done:
			done = False #I know this looks ridiculous: this prevents agent from falling out of the arena
			# return -1, self.bad_exit_reward, True, {}

		#ground_bot motion is random (same actions as drone)
		viable_next_states = []
		for gb_action in range(5):
			if gb_action == 0:
				if (ground_bot_row != (self.map_size[0]-1)) and not ((ground_bot_row == (self.map_size[0]-2)) and (ground_bot_col == (self.map_size[1]-1))):
					viable_next_states.append(self.encode_state(drone_row, drone_col, ground_bot_row+1, ground_bot_col))
			elif gb_action == 1:
				if (ground_bot_col != (self.map_size[1]-1)) and not ((ground_bot_row == (self.map_size[0]-1)) and (ground_bot_col == (self.map_size[1]-2))):
					viable_next_states.append(self.encode_state(drone_row, drone_col, ground_bot_row, ground_bot_col+1))
			elif gb_action == 2:
				if (ground_bot_row != 0) and not ((ground_bot_row == 1) and (ground_bot_col == 0)):
					viable_next_states.append(self.encode_state(drone_row, drone_col, ground_bot_row-1, ground_bot_col))
			elif gb_action == 3:
				if (ground_bot_col != 0) and not ((ground_bot_row == 0) and (ground_bot_col == 1)):
					viable_next_states.append(self.encode_state(drone_row, drone_col, ground_bot_row, ground_bot_col-1))
			elif gb_action == 4:
				viable_next_states.append(self.encode_state(drone_row, drone_col, ground_bot_row, ground_bot_col))

		next_state = viable_next_states[self.np_random.multinomial(1, np.ones(len(viable_next_states)) / float(len(viable_next_states))).argmax()]
		drone_row, drone_col, ground_bot_row, ground_bot_col = self.decode_state(next_state)
		observation = self.get_observation(drone_row, drone_col, ground_bot_row, ground_bot_col)

		if (drone_row == (self.map_size[0]-1)) and (drone_col == (self.map_size[1]-1)):
			done = True
			observation = -1
			reward = self.completion_reward
		elif (drone_row == ground_bot_row) and (drone_col == ground_bot_col):
			done = True
			observation = -1
			reward = self.position_match_reward
		else:
			reward = 0.

		self.current_state = next_state
		return observation, reward, done, False, {}

	def reset(self, seed=None, options=None):
		drone_row = self.init_pos[0]
		drone_col = self.init_pos[1]
		g_initlist = [i for i in range(self.map_size[0]*self.map_size[1] - 2)]
		g_initlist.remove(0)
		g_initlist.remove(self.map_size[1]-1)
		g_initlist.remove(self.map_size[1])

		g_encoding = g_initlist[self.np_random.multinomial(1, np.ones(len(g_initlist)) / float(len(g_initlist))).argmax()]
		ground_bot_col = (g_encoding+1) % self.map_size[1]
		ground_bot_row = (g_encoding+1) // self.map_size[1]
		self.start_state = self.encode_state(drone_row, drone_col, ground_bot_row, ground_bot_col)
		self.current_state = self.start_state
		observation = self.get_observation(drone_row, drone_col, ground_bot_row, ground_bot_col)
		return observation

	def render(self, mode=None):
		import pygame
		pygame.init()
		from pygame.draw import circle
		drone_row, drone_col, ground_bot_row, ground_bot_col = self.decode_state(self.current_state)

		size = cols, rows = 320, 320
		screen = pygame.display.set_mode(size)
		usable_cols = int(cols * 0.8)
		usable_rows = int(rows * 0.8)
		starting_col = (cols - usable_cols) // 2
		starting_row =  (rows - usable_rows) // 2
		row_steps = usable_rows // (self.map_size[0]-1)
		col_steps = usable_cols // (self.map_size[1]-1)

		white = (255, 255, 255)
		red = (255, 0, 0)
		blue = (0, 0, 255)
		black = (0, 0, 0)

		smalldot_size = 2
		mediumdot_size = 10
		largedot_size = 20

		screen.fill(white)

		circle(screen, blue, (starting_col + drone_col*col_steps, rows - (starting_row + drone_row*row_steps)), largedot_size) #plotting drone
		circle(screen, red, (starting_col + ground_bot_col*col_steps, rows - (starting_row + ground_bot_row*row_steps)), mediumdot_size) #plotting ground_bot
		for i in range(self.map_size[0]):
			for j in range(self.map_size[1]):
				circle(screen, black, (starting_col + j*col_steps, rows - (starting_row + i*row_steps)), smalldot_size) #plotting grid

		pygame.display.flip()

#environment tests

#1. encoding decoding test
# sum_stuff = 0
# DS = DroneSurveillanceEnv()
# obs_space = DS.state_space.n
# for i in range(obs_space):
# 	drone_row, drone_col, ground_bot_row, ground_bot_col = DS.decode_state(i)
# 	state = DS.encode_state(drone_row, drone_col, ground_bot_row, ground_bot_col)

# 	print(i, state, drone_row, drone_col, ground_bot_row, ground_bot_col)
# 	sum_stuff = sum_stuff + abs(i-state)

# print(sum_stuff)


#2. test basic render
# import time

# env = DroneSurveillanceEnv()
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

# env = DroneSurveillanceEnv()
# # obs = env.reset()

# #try with each action
# # actions = [4] * 100
# #complete path
# # actions = [0, 0, 0, 0, 1, 1, 1, 1]
# actions = [0, 1, 0, 1, 0, 1, 0, 1]

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