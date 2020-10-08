import torch
import torch.nn as nn
from torch.nn import ELU

#These networks are for reproducing low dimensional environment experiments
class rho_net_lowdim(nn.Module):
	def __init__(self, num_obs, num_actions, AIS_state_size = 5):
		super(rho_net_lowdim, self).__init__()
		input_ndims = num_obs + num_actions + 1
		self.AIS_state_size = AIS_state_size
		self.fc1 = nn.Linear(input_ndims, AIS_state_size)
		self.lstm1 = nn.LSTM(AIS_state_size, AIS_state_size)

	def forward(self, x, hidden=None):
		if hidden == None:
			hidden = (torch.randn(1, 1, self.AIS_state_size), torch.randn(1, 1, self.AIS_state_size))
		x = torch.tanh(self.fc1(x))
		x, hidden = self.lstm1(x, hidden)
		return x, hidden

#psi is the same as \hat P^y in the paper
class psi_net_lowdim(nn.Module):
	def __init__(self, num_obs, num_actions, AIS_state_size = 5):
		super(psi_net_lowdim, self).__init__()
		input_ndims = AIS_state_size + num_actions
		self.softmax = nn.Softmax(dim=0)
		self.fc1_r = nn.Linear(input_ndims, int(AIS_state_size/2))
		self.fc1_d = nn.Linear(input_ndims, int(AIS_state_size/2))
		self.fc2_r = nn.Linear(int(AIS_state_size/2), 1)
		self.fc2_d = nn.Linear(int(AIS_state_size/2), num_obs)

	def forward(self, x):
		x_r = torch.tanh(self.fc1_r(x))
		x_d = torch.tanh(self.fc1_d(x))
		reward = self.fc2_r(x_r)
		obs_probs = self.softmax(self.fc2_d(x_d))
		return reward, obs_probs

class policy_net_lowdim(nn.Module):
	def __init__(self, num_actions, AIS_state_size = 5, exploration_temp = 1.):
		super(policy_net_lowdim, self).__init__()
		self.exploration_temp = exploration_temp
		input_ndims = AIS_state_size
		self.softmax = nn.Softmax(dim=0)
		self.fc1 = nn.Linear(input_ndims, AIS_state_size)
		self.fc2 = nn.Linear(AIS_state_size, num_actions)

	def forward(self, x):
		x = torch.tanh(self.fc1(x))
		x = self.softmax(self.fc2(x) / self.exploration_temp)
		return x

#---------------------------------------------------------------------------------------------------

#These networks are for reproducing moderate dimensional environment experiments
class rho_net_moddim(nn.Module):
	def __init__(self, num_obs, num_actions, AIS_state_size = 5):
		super(rho_net_moddim, self).__init__()
		input_ndims = num_obs + num_actions + 1
		self.AIS_state_size = AIS_state_size
		self.lstm1 = nn.LSTM(input_ndims, AIS_state_size)

	def forward(self, x, hidden=None):
		if hidden == None:
			hidden = (torch.zeros(1, 1, self.AIS_state_size), torch.zeros(1, 1, self.AIS_state_size))
		x, hidden = self.lstm1(x, hidden)
		return x, hidden

#psi is the same as \hat P^y in the paper
class psi_net_moddim(nn.Module):
	def __init__(self, num_obs, num_actions, AIS_state_size = 5):
		super(psi_net_moddim, self).__init__()
		input_ndims = AIS_state_size + num_actions
		self.softmax = nn.Softmax(dim=0)
		self.fc1_r = nn.Linear(input_ndims, AIS_state_size//2)
		self.fc2_r = nn.Linear(AIS_state_size//2, 1)
		self.fc1_d = nn.Linear(input_ndims, AIS_state_size//2)
		self.fc2_d = nn.Linear(AIS_state_size//2, num_obs)

	def forward(self, x):
		x_r = torch.relu(self.fc1_r(x))
		x_d = torch.relu(self.fc1_d(x))
		reward = self.fc2_r(x_r)
		obs_probs = self.softmax(self.fc2_d(x_d))
		return reward, obs_probs

class policy_net_moddim(nn.Module):
	def __init__(self, num_actions, AIS_state_size = 5, exploration_temp = 1.):
		super(policy_net_moddim, self).__init__()
		self.exploration_temp = exploration_temp
		input_ndims = AIS_state_size
		self.softmax = nn.Softmax(dim=0)
		self.fc1 = nn.Linear(AIS_state_size, num_actions)

	def forward(self, x):
		x = self.softmax(self.fc1(x))
		return x

#---------------------------------------------------------------------------------------------------

#These networks are for reproducing high dimensional environment experiments
class rho_net_highdim(nn.Module):
	def __init__(self, obs_latent_space_size, num_actions, AIS_state_size = 5):
		super(rho_net_highdim, self).__init__()
		input_ndims = obs_latent_space_size + num_actions + 1
		self.AIS_state_size = AIS_state_size
		self.lstm1 = nn.LSTM(input_ndims, AIS_state_size)

	def forward(self, x, hidden=None):
		if hidden == None:
			hidden = (torch.zeros(1, 1, self.AIS_state_size), torch.zeros(1, 1, self.AIS_state_size))
		x, hidden = self.lstm1(x, hidden)
		return x, hidden

#psi is the same as \hat P^y in the paper
class psi_net_highdim_MMD(nn.Module):
	def __init__(self, obs_latent_space_size, num_actions, AIS_state_size = 5):
		super(psi_net_highdim_MMD, self).__init__()
		input_ndims = AIS_state_size + num_actions
		self.fc1_r = nn.Linear(input_ndims, AIS_state_size//2)
		self.fc1_d = nn.Linear(input_ndims, AIS_state_size//2)
		self.fc2_r = nn.Linear(AIS_state_size//2, 1)
		self.fc2_d = nn.Linear(AIS_state_size//2, obs_latent_space_size)

	def forward(self, x):
		x_r = torch.relu(self.fc1_r(x))
		x_d = torch.relu(self.fc1_d(x))
		reward = self.fc2_r(x_r)
		obs_probs = self.fc2_d(x_d)
		return reward, obs_probs

class psi_net_highdim_KL(nn.Module):
	def __init__(self, obs_latent_space_size, num_actions, AIS_state_size = 5, num_components = 5):
		super(psi_net_highdim_KL, self).__init__()
		input_ndims = AIS_state_size + num_actions
		self.eps = 1e-6
		self.num_components = num_components
		self.elu = ELU()
		self.softmax = nn.Softmax(dim=0)

		self.fc1_r = nn.Linear(input_ndims, AIS_state_size//2)
		self.fc2_r = nn.Linear(AIS_state_size//2, 1)

		self.fc1_d = nn.Linear(input_ndims, AIS_state_size//2)
		self.fc2_d_mean = nn.Linear(AIS_state_size//2, obs_latent_space_size*num_components)
		self.fc2_d_std = nn.Linear(AIS_state_size//2, obs_latent_space_size*num_components)
		self.fc2_d_mix = nn.Linear(AIS_state_size//2, num_components)

	def forward(self, x):
		x_r = torch.relu(self.fc1_r(x))
		reward = self.fc2_r(x_r)

		x_d = torch.relu(self.fc1_d(x))
		mvg_dist_mean = self.fc2_d_mean(x_d)
		mvg_dist_std = self.elu(self.fc2_d_std(x_d)) + 1. + self.eps
		mvg_dist_mix = self.softmax(self.fc2_d_mix(x_d))
		return reward, mvg_dist_mean.reshape(-1, self.num_components), mvg_dist_std.reshape(-1, self.num_components), mvg_dist_mix.reshape(-1, self.num_components)

class policy_net_highdim(nn.Module):
	def __init__(self, num_actions, AIS_state_size = 5, exploration_temp = 1.):
		super(policy_net_highdim, self).__init__()
		self.exploration_temp = exploration_temp
		input_ndims = AIS_state_size
		self.softmax = nn.Softmax(dim=0)
		self.fc1 = nn.Linear(input_ndims, AIS_state_size)
		self.fc2 = nn.Linear(AIS_state_size, num_actions)

	def forward(self, x):
		x = torch.relu(self.fc1(x))
		x = self.softmax(self.fc2(x))
		return x

#---------------------------------------------------------------------------------------------------