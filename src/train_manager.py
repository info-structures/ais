import os
import numpy as np
import torch
from torch.distributions import Categorical

from networks import rho_net, psi_net, policy_net

class batch_creator():
	def __init__(self, args, env, output_folder):
		self.args = args
		self.env = env
		self.AIS_SS = args.AIS_state_size #d_{\hat Z}

		self.rho = rho_net(obs_shape=self.env.observation_space['image'].shape, num_actions=self.env.action_space.n, AIS_state_size=self.AIS_SS)
		self.psi = psi_net(num_actions=self.env.action_space.n, AIS_state_size=self.AIS_SS, IPM=args.IPM, num_components=args.AIS_pred_ncomp)
		self.policy = policy_net(num_actions=self.env.action_space.n, AIS_state_size=self.AIS_SS)

		self.beta = args.beta #discount
		self.batch_size = args.batch_size
		self.seed = args.seed
		self.output_model_folder = os.path.join(output_folder, 'models')
		if not os.path.exists(self.output_model_folder):
			os.makedirs(self.output_model_folder)

		#the data of a single batch is stored in the following
		self.policy_history = torch.Tensor([])
		self.reward_episode = []
		self.pred_reward_episode = torch.Tensor([])
		self.next_aiss = torch.Tensor([])	
		self.next_ais_mean_estimates = torch.Tensor([])
		self.next_ais_std_estimates = torch.Tensor([])
		self.next_ais_mix_estimates = torch.Tensor([])
		self.aiss = torch.Tensor([])
		self.episode_id = torch.zeros(self.batch_size)

	def preprocess_minigrid_obs(self, obs):
		normalized_obs = obs.astype(np.float32)
		normalized_obs[:, :, 0] = normalized_obs[:, :, 0] / 10.0
		normalized_obs[:, :, 1] = normalized_obs[:, :, 1] / 5.0
		normalized_obs[:, :, 2] = normalized_obs[:, :, 2] / 2.0
		return torch.tensor(normalized_obs.flatten())

	def convert_int_to_onehot(self, value, num_values):
		onehot = torch.zeros(num_values)
		onehot[int(value)] = 1.
		return onehot

	def create_batch(self):
		self.policy_history = torch.Tensor([])
		self.reward_episode = []
		self.pred_reward_episode = torch.Tensor([])
		self.next_aiss = torch.Tensor([])
		self.next_ais_mean_estimates = torch.Tensor([])
		self.next_ais_std_estimates = torch.Tensor([])
		self.next_ais_mix_estimates = torch.Tensor([])
		self.aiss = torch.Tensor([])
		self.episode_id = torch.zeros(self.batch_size)

		num_samples = 0
		episode_counter = 0
		
		while num_samples < self.batch_size:
			hidden = None
			action = torch.zeros(self.env.action_space.n)
			reward = 0.

			current_obs = self.env.reset()['image']
			current_obs = self.preprocess_minigrid_obs(current_obs)

			for j in range(1000):
				rho_input = torch.cat((current_obs, action, torch.Tensor([reward]))).reshape(1, 1, -1)
				ais_z, hidden = self.rho(rho_input, hidden)
				ais_z = ais_z.reshape(-1)
				self.aiss = torch.cat((self.aiss, ais_z.detach().unsqueeze(0)))

				policy_probs = self.policy(ais_z)
				c = Categorical(policy_probs)
				action = c.sample()
				self.policy_history = torch.cat((self.policy_history, c.log_prob(action).unsqueeze(0)))

				next_obs, reward, done, _ = self.env.step(action.item())
				self.reward_episode.append(reward)

				action = self.convert_int_to_onehot(action, self.env.action_space.n)
				psi_input = torch.cat((ais_z, action))
				if self.args.IPM == 'KL':
					reward_est, next_ais_mean_est, next_ais_std_est, next_ais_mix_est = self.psi(psi_input)
					self.next_ais_mean_estimates = torch.cat((self.next_ais_mean_estimates, next_ais_mean_est.unsqueeze(0)))
					self.next_ais_std_estimates = torch.cat((self.next_ais_std_estimates, next_ais_std_est.unsqueeze(0)))
					self.next_ais_mix_estimates = torch.cat((self.next_ais_mix_estimates, next_ais_mix_est.unsqueeze(0)))
				elif self.args.IPM == "MMD":
					reward_est, next_aiss = self.psi(psi_input)
					self.next_aiss = torch.cat((self.next_aiss, next_aiss.unsqueeze(0)))
				else:
					assert False, "IPM should be MMD/KL. Given: {}".format(self.IPM)
				self.pred_reward_episode = torch.cat((self.pred_reward_episode, reward_est))

				current_obs = next_obs['image']
				current_obs = self.preprocess_minigrid_obs(current_obs)

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
		self.rho.eval()
		self.psi.eval()
		self.policy.eval()
		with torch.no_grad():
			returns = torch.Tensor([])
			for n_eps in range(n_episodes):
				reward_episode = []
				current_obs = self.env.reset()['image']
				current_obs = self.preprocess_minigrid_obs(current_obs)
				action = torch.zeros(self.env.action_space.n)
				reward = 0.
				hidden = None
				for j in range(1000):
					rho_input = torch.cat((current_obs, action, torch.Tensor([reward]))).reshape(1, 1, -1)
					ais_z, hidden = self.rho(rho_input, hidden)
					ais_z = ais_z.reshape(-1)

					policy_probs = self.policy(ais_z)
					if greedy:
						c = Categorical(policy_probs)
						_, action = policy_probs.max(0)
					else:
						c = Categorical(policy_probs)
						action = c.sample()
					next_obs, reward, done, _ = self.env.step(action.item())

					action = self.convert_int_to_onehot(action, self.env.action_space.n)
					reward_episode.append(reward)

					current_obs = next_obs['image']
					current_obs = self.preprocess_minigrid_obs(current_obs)

					if done:
						break

				rets = []
				R = 0
				for i, r in enumerate(reward_episode[::-1]):
					R = r + self.beta * R
					rets.insert(0,R)
				rets = torch.Tensor(rets)
				returns = torch.cat((returns, rets[0].reshape(-1)))

		self.rho.train()
		self.psi.train()
		self.policy.train()

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