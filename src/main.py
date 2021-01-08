import os
import numpy as np
import argparse
import gym
import gym_minigrid
from gym_minigrid.wrappers import *
from train_manager import batch_creator

import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn.modules import MSELoss as MSELoss
from torch.distributions import Categorical
from torch.distributions.multivariate_normal import MultivariateNormal
from torch.utils.tensorboard import SummaryWriter

parser = argparse.ArgumentParser(description='This code runs AIS using the Next Observation prediction version')

parser.add_argument("--output_dir", help="Directory to store output results in", default="results")
parser.add_argument("--env_name", help="Gym Environment to Use. Options: `Tiger-v0`, `Voicemail-v0`, `CheeseMaze-v0`\n `DroneSurveillance-v0`, `RockSampling-v0`", default="Tiger-v0")
parser.add_argument("--eval_frequency", type=int, help="Number of Batch Iterations per Evaluation", default=100)
parser.add_argument("--N_eps_eval", type=int, help="Number of Episodes per Evaluation", default=50)
parser.add_argument("--beta", type=float, help="Discount Factor", default=0.95)
parser.add_argument("--lmbda",	type=float, help="lambda value (Trade off between next reward and next observation prediction)", default=0.0001)
parser.add_argument("--policy_LR", type=float, help="Learning Rate for Policy Network", default=0.0007)
parser.add_argument("--ais_LR",	type=float, help="Learning Rate for AIS Netowrk", default=0.003)
parser.add_argument("--batch_size", type=int, help="Number of samples per batch of training", default=200)
parser.add_argument("--num_batches", type=int, help="Number of batches used in training", default=2000)
parser.add_argument("--AIS_state_size", type=int, help="Size of the AIS vector in the neural network", default=40)
parser.add_argument("--IPM", help="Options: `KL`, `MMD`", default='MMD')
parser.add_argument("--seed", type=int, help="Random seed of the experiment", default=42)
parser.add_argument("--models_folder", type=str, help='Pretrained model (state dict)')
parser.add_argument("--AIS_pred_ncomp", type=int, help="Number of Components used in the GMM to predict next AIS (For MiniGrid+KL)", default=5)

args = parser.parse_args()

env = gym.make(args.env_name)
eval_frequency = args.eval_frequency
N_eps_eval = args.N_eps_eval
beta = args.beta #beta is the discount factor
lmbda = args.lmbda
policy_LR = args.policy_LR
ais_LR = args.ais_LR
batch_size = args.batch_size
num_batches = args.num_batches
AIS_SS = args.AIS_state_size #d_{\hat Z}
AIS_PN = args.AIS_pred_ncomp #this arg is only used for MiniGrid with the KL IPM
IPM = args.IPM
seed = args.seed

#set random seeds
np.random.seed(seed)
torch.manual_seed(seed)
env.seed(seed)

if (args.env_name[:8] == 'MiniGrid') and (args.IPM == 'KL'):
	args_list = [str(args.env_name), str(eval_frequency), str(N_eps_eval), str(beta), str(lmbda), str(policy_LR), str(ais_LR), str(batch_size), str(num_batches), str(AIS_SS), str(AIS_PN), str(IPM)]
else:
	args_list = [str(args.env_name), str(eval_frequency), str(N_eps_eval), str(beta), str(lmbda), str(policy_LR), str(ais_LR), str(batch_size), str(num_batches), str(AIS_SS), str(IPM)]
output_folder = os.path.join(args.output_dir , '_'.join(args_list))
if not os.path.exists(output_folder):
	try:
		os.makedirs(output_folder)
	except:
		pass

def eval_and_save_stuff(_bc, _N_eps_eval, _batch_num, _perf_array, policy_optimizer, AIS_optimizer):
	performance = _bc.eval_performance(greedy=False, n_episodes=_N_eps_eval)
	writer.add_scalar("Performance/" + _bc.args.env_name + "/performance", performance, _batch_num)
	print ('Performance on Iteration No.', _batch_num, ': ', performance)
	_perf_array.append(performance)

	_bc.save_networks()
	#save optimizer state to continue training at a later stage
	torch.save(policy_optimizer.state_dict(), os.path.join(_bc.output_model_folder, 'policy_optimizer.pth'))
	torch.save(AIS_optimizer.state_dict(), os.path.join(_bc.output_model_folder, 'AIS_optimizer.pth'))
	return

if __name__ == "__main__":
	bc = batch_creator(args, env, output_folder)
	writer = SummaryWriter()

	# Use eqn 60/61, do not fit observation, fit AIS
	fit_obs = False

	policy_optimizer = optim.Adam(bc.policy.parameters(), lr=policy_LR)
	AIS_optimizer = optim.Adam(list(bc.rho.parameters()) + list(bc.psi.parameters()), lr=ais_LR)

	if args.models_folder:
		#loads model neural networks as well as optimizer state to continue training
		bc.load_models_from_folder(args.models_folder)
		policy_optimizer.load_state_dict(torch.load(os.path.join(args.models_folder, 'policy_optimizer.pth')))
		AIS_optimizer.load_state_dict(torch.load(os.path.join(args.models_folder, 'AIS_optimizer.pth')))

	perf_array = []
	for batch_num in range(num_batches):
		if ((batch_num) % eval_frequency == 0) or (batch_num == 0):
			eval_and_save_stuff(bc, N_eps_eval, batch_num, perf_array, policy_optimizer, AIS_optimizer)
		bc.create_batch()

		#reinforce policy gradient update (backward view implemented here)
		returns = []
		current_eid = -1
		for i, r in enumerate(bc.reward_episode[::-1]):
			if current_eid != bc.episode_id[batch_size - i - 1]:
				R = 0
				current_eid = bc.episode_id[batch_size - i - 1]
			R = r + beta * R
			returns.insert(0,R)
		returns = torch.Tensor(returns)
		policy_loss = torch.sum(torch.mul(bc.policy_history, returns).mul(-1), -1)

		#update (\hat \rho) and (\hat P^y)
		mse_loss = MSELoss()
		rewards = torch.Tensor(bc.reward_episode)
		reward_loss = mse_loss(rewards, bc.pred_reward_episode)

		seg_first = 0
		current_eid = 0
		next_obs_loss = 0.
		count = 0

		for i in range(bc.observations.shape[0]):
			if bc.episode_id[i] == current_eid:
				continue
			else: #this condition is triggered at the transitions between episodes in a batch
				seg_last = i-1
				#this for loop processes a single episode in a batch
				for j in range(seg_first+1, seg_last+1):
					if IPM == 'MMD': #this expression is obtained by considering L2 norm squared for the kernel based IPM metric
						next_obs_loss += 2*torch.norm(bc.obs_probs[j-1, :])**2 - 4*torch.dot(bc.obs_probs[j-1, :], bc.observations[j])
					elif IPM == 'KL': #this uses KL to upper-bound Wasserstein distance (which is an IPM)
						if args.env_name[:8] == 'MiniGrid':
							mixture_probs = torch.Tensor([])
							for d in range(0, bc.mvg_dist_mean_estimates.shape[2]):
								m = MultivariateNormal(bc.mvg_dist_mean_estimates[j-1, :, d], torch.diag(bc.mvg_dist_std_estimates[j-1, :, d]))
								if bc.mvg_dist_mix_estimates[j-1, 0, d] != 0.0:
									mixture_probs = torch.cat((mixture_probs, (torch.log(bc.mvg_dist_mix_estimates[j-1, 0, d]) + m.log_prob(bc.observations[j, :])).unsqueeze(0)))
							next_obs_loss = next_obs_loss - torch.logsumexp(mixture_probs, dim=0)
						else:
							m = Categorical(bc.obs_probs[j-1, :])
							if fit_obs:
								next_obs_loss = next_obs_loss - m.log_prob(bc.observations[j, :].argmax())
							else:
								next_obs_loss = next_obs_loss - m.log_prob(bc.ais[j, :].argmax())
					count = count + 1
				current_eid = current_eid + 1
				seg_first = i
		seg_last = i

		#this for loop accounts for the last episode in the batch
		for j in range(seg_first+1, seg_last+1):
			if IPM == 'MMD': #this expression is obtained by considering L2 norm squared for the kernel based IPM metric
				next_obs_loss += 2*torch.norm(bc.obs_probs[j-1, :])**2 - 4*torch.dot(bc.obs_probs[j-1, :], bc.observations[j])
			elif IPM == 'KL': #this uses KL to upper-bound Wasserstein distance (which is an IPM)
				if args.env_name[:8] == 'MiniGrid':
					mixture_probs = torch.Tensor([])
					for d in range(0, bc.mvg_dist_mean_estimates.shape[2]):
						m = MultivariateNormal(bc.mvg_dist_mean_estimates[j-1, :, d], torch.diag(bc.mvg_dist_std_estimates[j-1, :, d]))
						if bc.mvg_dist_mix_estimates[j-1, 0, d] != 0.0:
							mixture_probs = torch.cat((mixture_probs, (torch.log(bc.mvg_dist_mix_estimates[j-1, 0, d]) + m.log_prob(bc.observations[j, :])).unsqueeze(0)))
					next_obs_loss = next_obs_loss - torch.logsumexp(mixture_probs, dim=0)
				else:
					m = Categorical(bc.obs_probs[j-1, :])
					if fit_obs:
						next_obs_loss = next_obs_loss - m.log_prob(bc.observations[j, :].argmax())
					else:
						next_obs_loss = next_obs_loss - m.log_prob(bc.ais[j, :].argmax())
			count = count + 1

		if count != 0:
			next_obs_loss = next_obs_loss / float(count)

		#add all losses together and do a single backprop from total_loss
		total_loss = lmbda * reward_loss + (1-lmbda)*next_obs_loss + policy_loss

		policy_optimizer.zero_grad()
		AIS_optimizer.zero_grad()
		total_loss.backward()
		AIS_optimizer.step()
		policy_optimizer.step()
		# performance = bc.eval_performance(greedy=False, n_episodes=N_eps_eval)
		# writer.add_scalar("Performance/performance", performance, batch_num)

	writer.flush()
	eval_and_save_stuff(bc, N_eps_eval, batch_num, perf_array, policy_optimizer, AIS_optimizer)

	output_filename = 'seed' + str(seed) + '.npz'
	np.savez(os.path.join(output_folder, output_filename), perf_array = np.array(perf_array))
