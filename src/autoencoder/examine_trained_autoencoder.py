from simple_autoencoder import autoencoder, unNormalize

import os
import argparse
import time

import torch
from torch import nn
from torch.utils.data import DataLoader

from minigrid_datasets import MGER6x6, MGMRN2S4, MGDK6x6, MGER8x8, MGDK8x8, MGFR


LOSS_TOL = 1e-2

parser = argparse.ArgumentParser()

parser.add_argument(
    "--dataset_name",
    help="Name of the Dataset based on Task",
    default='MGER6x6'
)

parser.add_argument(
    "--dataset_folder",
    help="Folder where the rollout data is collected for training the autoencoder",
    default='rollout_data/MGER6x6'
)

parser.add_argument(
    "--batch_size",
    type=int,
    help="Batch Size over dataset",
    default=100
)

parser.add_argument(
    "--model_file",
    help="File to examine autoencoded outputs compared to inputs",
    default='models/MGER6x6/LatentSize16_withLargeStd_withLowLR_WD/autoencoder_final.pth'
)

args = parser.parse_args()

criterion = nn.MSELoss()

batch_size = args.batch_size

big_obs_grid = False
view_row = 3
view_col = 3
if args.dataset_name == 'MGER6x6':
    dataset = MGER6x6(args.dataset_folder)
elif args.dataset_name == 'MGMRN2S4':
    dataset = MGMRN2S4(args.dataset_folder)
elif args.dataset_name == 'MGDK6x6':
    dataset = MGDK6x6(args.dataset_folder)
else:
    big_obs_grid = True
    view_row = 7
    view_col = 7
    if args.dataset_name == 'MGER8x8':
        dataset = MGER8x8(args.dataset_folder)
    elif args.dataset_name == 'MGDK8x8':
        dataset = MGDK8x8(args.dataset_folder)
    elif args.dataset_name == 'MGFR':
        dataset = MGFR(args.dataset_folder)

dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

model = autoencoder(big_obs_grid)

model.load_state_dict(torch.load(args.model_file))

for data in dataloader:
	output = model(data)
	# loss = criterion(output, data).item()
    if args.dataset_name == 'MGER6x6' or args.dataset_name == 'MGMRN2S4' or args.dataset_name == 'MGDK6x6':
        unNormalized_data = unNormalize(data, dataset.mean, dataset.std, view_row, view_col)
    else:
        unNormalized_data = unNormalize(data, dataset.mean, dataset.max_vals, view_row, view_col)
	output_display = output.reshape(-1)
	data_display = data.reshape(-1)
	unNormalized_data_display = unNormalized_data.reshape(-1)

	# print('LOSS IS ------------------------------------------------>', loss)
	# print ('Data    |    Output')
	for i, x in enumerate(output_display):
		loss = criterion(data_display[i], output_display[i]).item()
		if loss >= LOSS_TOL:
			print (unNormalized_data_display[i].item(), data_display[i].item(), output_display[i].item(), loss)
	# print(output.reshape(-1), data.reshape(-1))
	time.sleep(1)