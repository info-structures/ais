import os
import argparse
import numpy as np
import torch

import gym
import gym_minigrid
from gym_minigrid.wrappers import *

from random import randint

STD_TOL = 1e-9

parser = argparse.ArgumentParser()

parser.add_argument(
    "--env",
    help="gym environment to load",
    default='MiniGrid-Empty-Random-6x6-v0'
)
parser.add_argument(
    "--agent_view_size",
    type=int,
    help="Grid Size that the agent sees"
)
parser.add_argument(
    "--num_episodes",
    type=int,
    help="Number of random rollouts to perform",
    default=1000
)
parser.add_argument(
    "--save_folder",
    help="folder to save data in",
    default='autoencoder/rollout_data/MGER6x6'
)

args = parser.parse_args()

if not os.path.exists(args.save_folder):
	os.makedirs(args.save_folder)

env = gym.make(args.env)
if args.agent_view_size is not None:
    env = ViewSizeWrapper(env, agent_view_size=args.agent_view_size)

num_actions = env.action_space.n

img_db = np.expand_dims(env.reset()['image'], axis=0)


print (img_db.shape)
for n_episode in range(args.num_episodes):
	print ('Running episode {}'.format(n_episode))
	done = False
	obs = env.reset()
	temp = np.expand_dims(obs['image'], axis=0)
	while not done:
		action = randint(0, num_actions-1)
		obs, reward, done, info = env.step(action)
		if not done:
			temp = np.concatenate((temp, np.expand_dims(obs['image'], axis=0)), axis=0)
	img_db = np.concatenate((img_db, temp), axis=0)


# img_db = torch.as_tensor(img_db)
torch.save(img_db, os.path.join(args.save_folder, 'training.pt'))
print(img_db.shape)

img_db_float = torch.Tensor([img_db[0:100000, :]]).reshape(-1, 3) / 255.0
mean = torch.mean(img_db_float, dim=0)
std = torch.std(img_db_float, dim=0) + STD_TOL
max_vals = torch.max(img_db_float, dim=0)[0] + STD_TOL

print(mean)
print(std)
print(max_vals)

torch.save(mean, os.path.join(args.save_folder, 'mean.pt'))
torch.save(std, os.path.join(args.save_folder, 'std.pt'))
torch.save(max_vals, os.path.join(args.save_folder, 'max_vals.pt'))