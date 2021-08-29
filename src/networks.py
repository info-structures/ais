import torch
import torch.nn as nn
from torch.nn import ELU

import numpy as np

class rho_net(nn.Module):
	def __init__(self, obs_shape, num_actions, AIS_state_size):
		super(rho_net, self).__init__()
		input_ndims = np.prod(obs_shape) + num_actions + 1
		self.AIS_state_size = AIS_state_size
		self.lstm1 = nn.LSTM(input_ndims, AIS_state_size)

	def forward(self, x, hidden=None):
		if hidden == None:
			hidden = (torch.zeros(1, 1, self.AIS_state_size), torch.zeros(1, 1, self.AIS_state_size))
		x, hidden = self.lstm1(x, hidden)
		return x, hidden

class psi_net(nn.Module):
	def __init__(self, num_actions, AIS_state_size, IPM="MMD", num_components=-1):
		super(psi_net, self).__init__()
		self.num_actions = num_actions
		self.AIS_state_size = AIS_state_size
		self.IPM = IPM
		
		input_ndims = self.AIS_state_size + self.num_actions
		self.fc1_r = nn.Linear(input_ndims, AIS_state_size)
		self.fc2_r = nn.Linear(AIS_state_size, 1)

		self.fc1_d = nn.Linear(input_ndims, AIS_state_size)
		if self.IPM == "MMD":
			self.fc2_d = nn.Linear(AIS_state_size, AIS_state_size)
		elif self.IPM == "KL":
			self.num_components = num_components
			self.eps = 1e-6
			self.softmax = nn.Softmax(dim=0)
			self.elu = ELU()
			
			self.fc2_d_mean = nn.Linear(AIS_state_size, AIS_state_size*num_components)
			self.fc2_d_std = nn.Linear(AIS_state_size, AIS_state_size*num_components)
			self.fc2_d_mix = nn.Linear(AIS_state_size, num_components)
		else:
			assert False, "IPM should be MMD/KL. Given: {}".format(self.IPM)

	def forward(self, x):
		x_r = torch.relu(self.fc1_r(x))
		reward = self.fc2_r(x_r)

		x_d = torch.relu(self.fc1_d(x))
		if self.IPM == "MMD":
			next_ais = self.fc2_d(x_d)
			return reward, next_ais
		elif self.IPM == "KL":
			next_ais_mean = self.fc2_d_mean(x_d)
			next_ais_std = self.elu(self.fc2_d_std(x_d)) + 1. + self.eps
			next_ais_mix = self.softmax(self.fc2_d_mix(x_d))
			return reward, next_ais_mean.reshape(-1, self.num_components), next_ais_std.reshape(-1, self.num_components), next_ais_mix.reshape(-1, self.num_components)
		else:
			assert False, "IPM should be MMD/KL. Given: {}".format(self.IPM)

class policy_net(nn.Module):
	def __init__(self, num_actions, AIS_state_size, exploration_temp=1.0):
		super(policy_net, self).__init__()
		self.exploration_temp = exploration_temp
		
		self.fc1 = nn.Linear(AIS_state_size, AIS_state_size)
		self.fc2 = nn.Linear(AIS_state_size, num_actions)

		self.softmax = nn.Softmax(dim=0)

	def forward(self, x):
		x = torch.relu(self.fc1(x))
		x = self.softmax(self.fc2(x))
		return x