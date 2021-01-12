import os
import numpy as np
import torch
from torch.distributions import Categorical

from torchvision import transforms
from PIL import Image
from autoencoder.simple_autoencoder import autoencoder

from networks import rho_net_lowdim, psi_net_lowdim, policy_net_lowdim
from networks import rho_net_moddim, psi_net_moddim, policy_net_moddim
from networks import rho_net_highdim, psi_net_highdim_MMD, psi_net_highdim_KL, policy_net_highdim

class batch_creator():
	def __init__(self, args, env, output_folder, fit_obs):
		self.args = args
		self.env = env
		self.AIS_SS = args.AIS_state_size #d_{\hat Z}

		#different networks used for different class of environments
		#for small and finite action and observation spaces, lowdim or moddim can be used
		if args.env_name == "Tiger-v0" or args.env_name == "Voicemail-v0" or args.env_name == "CheeseMaze-v0":
			self.rho = rho_net_lowdim(num_obs = self.env.observation_space.n, num_actions = self.env.action_space.n, AIS_state_size = self.AIS_SS)
			if fit_obs:
				self.psi = psi_net_lowdim(num_obs = self.env.observation_space.n, num_actions = self.env.action_space.n, AIS_state_size = self.AIS_SS)
			else:
				self.psi = psi_net_lowdim(num_obs=self.AIS_SS, num_actions=self.env.action_space.n,
										  AIS_state_size=self.AIS_SS)
			self.policy = policy_net_lowdim(num_actions = self.env.action_space.n, AIS_state_size = self.AIS_SS)
		elif args.env_name == "DroneSurveillance-v0" or args.env_name == "RockSampling-v0":
			self.rho = rho_net_moddim(num_obs = self.env.observation_space.n, num_actions = self.env.action_space.n, AIS_state_size = self.AIS_SS)
			self.psi = psi_net_moddim(num_obs = self.env.observation_space.n, num_actions = self.env.action_space.n, AIS_state_size = self.AIS_SS)
			self.policy = policy_net_moddim(num_actions = self.env.action_space.n, AIS_state_size = self.AIS_SS)
		elif args.env_name[:8] == 'MiniGrid':
			autoencoder_folder = os.path.join('src', 'autoencoder', args.env_name)
			self.autoencoder_model = autoencoder(True)
			self.autoencoder_model.load_state_dict(torch.load(os.path.join(autoencoder_folder, 'autoencoder_final.pth'), map_location=torch.device('cpu')))
			self.observation_mean = torch.load(os.path.join(autoencoder_folder, 'mean.pt'))
			self.observation_scaler = torch.load(os.path.join(autoencoder_folder, 'max_vals.pt'))*1.2
			self.transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(self.observation_mean, self.observation_scaler)])

			self.rho = rho_net_highdim(obs_latent_space_size = self.autoencoder_model.latent_space_size, num_actions = self.env.action_space.n, AIS_state_size = self.AIS_SS)
			if args.IPM == 'MMD':
				self.psi = psi_net_highdim_MMD(obs_latent_space_size = self.autoencoder_model.latent_space_size, num_actions = self.env.action_space.n, AIS_state_size = self.AIS_SS)
			elif args.IPM == 'KL':
				self.psi = psi_net_highdim_KL(obs_latent_space_size = self.autoencoder_model.latent_space_size, num_actions = self.env.action_space.n, AIS_state_size = self.AIS_SS, num_components = args.AIS_pred_ncomp)
			self.policy = policy_net_highdim(num_actions = self.env.action_space.n, AIS_state_size = self.AIS_SS)

		self.beta = args.beta #discount
		self.batch_size = args.batch_size
		self.seed = args.seed
		if output_folder is not None:
			self.output_model_folder = os.path.join(output_folder, 'models', 'seed' + str(self.seed))
			if not os.path.exists(self.output_model_folder):
				os.makedirs(self.output_model_folder)

		#the data of a single bacth is stored in the following
		self.policy_history = torch.Tensor([])
		self.reward_episode = []
		self.pred_reward_episode = torch.Tensor([])
		self.obs_probs = torch.Tensor([])	
		self.mvg_dist_mean_estimates = torch.Tensor([])
		self.mvg_dist_std_estimates = torch.Tensor([])
		self.mvg_dist_mix_estimates = torch.Tensor([])
		self.observations = torch.Tensor([])
		self.episode_id = torch.zeros(self.batch_size)

	def convert_int_to_onehot(self, value, num_values):
		onehot = torch.zeros(num_values)
		onehot[int(value)] = 1.
		return onehot

	def get_encoded_obs(self, obs):
		obs = Image.fromarray(obs)
		obs = self.transform(obs)
		obs = obs.reshape(-1)
		encoded_obs = self.autoencoder_model(obs, getLatent=True).detach()
		return encoded_obs

	def create_batch(self):
		self.policy_history = torch.Tensor([])
		self.reward_episode = []
		self.pred_reward_episode = torch.Tensor([])
		self.obs_probs = torch.Tensor([])
		self.mvg_dist_mean_estimates = torch.Tensor([])
		self.mvg_dist_std_estimates = torch.Tensor([])
		self.mvg_dist_mix_estimates = torch.Tensor([])
		self.observations = torch.Tensor([])
		self.ais = torch.Tensor([])
		self.episode_id = torch.zeros(self.batch_size)

		num_samples = 0
		episode_counter = 0
		
		while num_samples < self.batch_size:
			hidden = None
			action = torch.zeros(self.env.action_space.n)
			reward = 0.
			if self.args.env_name[:8] == 'MiniGrid':
				current_obs = self.env.reset()['image']
				current_obs = self.get_encoded_obs(current_obs)
			else:
				current_obs = self.env.reset()
				current_obs = self.convert_int_to_onehot(current_obs, self.env.observation_space.n)

			for j in range(1000):
				self.observations = torch.cat((self.observations, current_obs.unsqueeze(0)))
				rho_input = torch.cat((current_obs, action, torch.Tensor([reward]))).reshape(1, 1, -1)
				ais_z, hidden = self.rho(rho_input, hidden)
				ais_z = ais_z.reshape(-1)
				self.ais = torch.cat((self.ais, ais_z.unsqueeze(0)))

				policy_probs = self.policy(ais_z.detach())

				c = Categorical(policy_probs)
				action = c.sample()

				self.policy_history = torch.cat((self.policy_history, c.log_prob(action).unsqueeze(0)))

				next_obs, reward, done, _ = self.env.step(action.item())
				self.reward_episode.append(reward)

				action = self.convert_int_to_onehot(action, self.env.action_space.n)
				psi_input = torch.cat((ais_z, action))
				if (self.args.env_name[:8] == 'MiniGrid') and (self.args.IPM == 'KL'):
					reward_est, mvg_dist_mean_est, mvg_dist_std_est, mvg_dist_mix_est = self.psi(psi_input)
					self.mvg_dist_mean_estimates = torch.cat((self.mvg_dist_mean_estimates, mvg_dist_mean_est.unsqueeze(0)))
					self.mvg_dist_std_estimates = torch.cat((self.mvg_dist_std_estimates, mvg_dist_std_est.unsqueeze(0)))
					self.mvg_dist_mix_estimates = torch.cat((self.mvg_dist_mix_estimates, mvg_dist_mix_est.unsqueeze(0)))
				else:
					reward_est, obs_probs = self.psi(psi_input)
					self.obs_probs = torch.cat((self.obs_probs, obs_probs.unsqueeze(0)))
				self.pred_reward_episode = torch.cat((self.pred_reward_episode, reward_est))

				if self.args.env_name[:8] == 'MiniGrid':
					current_obs = next_obs['image']
					current_obs = self.get_encoded_obs(current_obs)
				else:
					current_obs = next_obs
					current_obs = self.convert_int_to_onehot(current_obs, self.env.observation_space.n)

				self.episode_id[int(num_samples)] = int(episode_counter)
				num_samples = num_samples + 1

				if done:
					episode_counter = episode_counter + 1
					break
				if num_samples >= self.batch_size:
					break
		self.env.close()

		return

	#evaluate the current policy over `n_episodes` independent episodes
	def eval_performance(self, greedy=False, n_episodes=1):
		with torch.no_grad():
			returns = torch.Tensor([])
			for n_eps in range(n_episodes):
				reward_episode = []

				if self.args.env_name[:8] == 'MiniGrid':
					current_obs = self.env.reset()['image']
					current_obs = self.get_encoded_obs(current_obs)
				else:
					current_obs = self.env.reset()
					current_obs = self.convert_int_to_onehot(current_obs, self.env.observation_space.n)
				action = torch.zeros(self.env.action_space.n)
				reward = 0.
				hidden = None

				for j in range(1000):
					rho_input = torch.cat((current_obs, action, torch.Tensor([reward]))).reshape(1, 1, -1)
					ais_z, hidden = self.rho(rho_input, hidden)
					ais_z = ais_z.reshape(-1)

					policy_probs = self.policy(ais_z.detach())
					if greedy:
						c = Categorical(policy_probs)
						_, action = policy_probs.max(0)
					else:
						c = Categorical(policy_probs)
						action = c.sample()
					next_obs, reward, done, _ = self.env.step(action.item())

					action = self.convert_int_to_onehot(action, self.env.action_space.n)
					reward_episode.append(reward)

					if self.args.env_name[:8] == 'MiniGrid':
						current_obs = next_obs['image']
						current_obs = self.get_encoded_obs(current_obs)
					else:
						current_obs = next_obs
						current_obs = self.convert_int_to_onehot(current_obs, self.env.observation_space.n)

					if done:
						break

				rets = []
				R = 0
				for i, r in enumerate(reward_episode[::-1]):
					R = r + self.beta * R
					rets.insert(0,R)
				rets = torch.Tensor(rets)
				returns = torch.cat((returns, rets[0].reshape(-1)))

		return torch.mean(returns).item()

	def save_networks(self):
		torch.save(self.rho.state_dict(), os.path.join(self.output_model_folder, 'rho.pth'))
		torch.save(self.psi.state_dict(), os.path.join(self.output_model_folder, 'psi.pth'))
		torch.save(self.policy.state_dict(), os.path.join(self.output_model_folder, 'policy.pth'))
		return

	def load_models_from_folder(self, models_folder):
		self.rho.load_state_dict(torch.load(os.path.join(models_folder, 'rho.pth')))
		self.psi.load_state_dict(torch.load(os.path.join(models_folder, 'psi.pth')))
		self.policy.load_state_dict(torch.load(os.path.join(models_folder, 'policy.pth')))
		return