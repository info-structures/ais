import os
import numpy as np
import torch
import torch.utils.data as data
from torchvision import transforms

from PIL import Image


# ------------------------------------------------ 3 x 3 Observation grids ------------------------------------------------

#MiniGrid-Empty-Random-6x6-v0
class MGER6x6(data.Dataset):
	def __init__(self, data_folder):
		self.data_folder = data_folder
		self.training_file = 'training.pt'
		self.data = torch.load(os.path.join(self.data_folder, self.training_file))
		self.mean = torch.load(os.path.join(self.data_folder, 'mean.pt'))
		self.std = torch.load(os.path.join(self.data_folder, 'std.pt'))*8
		self.transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(self.mean, self.std)])
		# print (self.mean)
		# print (self.std)

	def __getitem__(self, index):
		data_item = self.data[index]
		# print (data_item.reshape(-1))
		# normalized_DI = torch.Tensor(data_item) / 255.0

		# for i in range(3):
		# 	normalized_DI[:,:,i] = (normalized_DI[:,:,i] - self.mean[i]) / self.std[i]
		# print(normalized_DI)

		# print (data_item.reshape(-1))
		# exit()
		# print (self.data.dtype)
		# exit()
		data_item = Image.fromarray(data_item)
		# print(list(data_item.getdata()))
		# exit()
		if self.transform is not None:
			data_item = self.transform(data_item)
		# print(list(data_item.getdata()))
		# exit()
		# print (data_item)
		# print (data_item.shape)
		# exit()
		return data_item.reshape(-1)

	def __len__(self):
		return len(self.data)


#MiniGrid-MultiRoom-N2-S4-v0
class MGMRN2S4(data.Dataset):
	def __init__(self, data_folder):
		self.data_folder = data_folder
		self.training_file = 'training.pt'
		self.data = torch.load(os.path.join(self.data_folder, self.training_file))
		self.mean = torch.load(os.path.join(self.data_folder, 'mean.pt'))
		self.std = torch.load(os.path.join(self.data_folder, 'std.pt'))*8
		self.transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(self.mean, self.std)])
		# print (self.mean)
		# print (self.std)

	def __getitem__(self, index):
		data_item = self.data[index]
		# print (data_item.reshape(-1))
		# normalized_DI = torch.Tensor(data_item) / 255.0

		# for i in range(3):
		# 	normalized_DI[:,:,i] = (normalized_DI[:,:,i] - self.mean[i]) / self.std[i]
		# print(normalized_DI)

		# print (data_item.reshape(-1))
		# exit()
		# print (self.data.dtype)
		# exit()
		data_item = Image.fromarray(data_item)
		# print(list(data_item.getdata()))
		# exit()
		if self.transform is not None:
			data_item = self.transform(data_item)
		# print(list(data_item.getdata()))
		# exit()
		# print (data_item)
		# print (data_item.shape)
		# exit()
		return data_item.reshape(-1)

	def __len__(self):
		return len(self.data)



#MiniGrid-DoorKey-6x6-v0
class MGDK6x6(data.Dataset):
	def __init__(self, data_folder):
		self.data_folder = data_folder
		self.training_file = 'training.pt'
		self.data = torch.load(os.path.join(self.data_folder, self.training_file))
		self.mean = torch.load(os.path.join(self.data_folder, 'mean.pt'))
		self.std = torch.load(os.path.join(self.data_folder, 'std.pt'))*8
		self.transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(self.mean, self.std)])
		# print (self.mean)
		# print (self.std)

	def __getitem__(self, index):
		data_item = self.data[index]
		# print (data_item.reshape(-1))
		# normalized_DI = torch.Tensor(data_item) / 255.0

		# for i in range(3):
		# 	normalized_DI[:,:,i] = (normalized_DI[:,:,i] - self.mean[i]) / self.std[i]
		# print(normalized_DI)

		# print (data_item.reshape(-1))
		# exit()
		# print (self.data.dtype)
		# exit()
		data_item = Image.fromarray(data_item)
		# print(list(data_item.getdata()))
		# exit()
		if self.transform is not None:
			data_item = self.transform(data_item)
		# print(list(data_item.getdata()))
		# exit()
		# print (data_item)
		# print (data_item.shape)
		# exit()
		return data_item.reshape(-1)

	def __len__(self):
		return len(self.data)



# ------------------------------------------------ 7 x 7 Observation grids ------------------------------------------------

#Parent Class for 7x7 Observation Grids
class ObsGrids7x7(data.Dataset):
	def __init__(self, data_folder):
		self.data_folder = data_folder
		self.training_file = 'training.pt'
		self.data = torch.load(os.path.join(self.data_folder, self.training_file))
		self.mean = torch.load(os.path.join(self.data_folder, 'mean.pt'))
		self.max_vals = torch.load(os.path.join(self.data_folder, 'max_vals.pt'))*1.2
		self.transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(self.mean, self.max_vals)])

	def __getitem__(self, index):
		data_item = self.data[index]
		data_item = Image.fromarray(data_item)
		if self.transform is not None:
			data_item = self.transform(data_item)
		return data_item.reshape(-1)

	def __len__(self):
		return len(self.data)


# Initially tried environments

#MiniGrid-Empty-8x8-v0
class MGER8x8(ObsGrids7x7):
	pass

#MiniGrid-DoorKey-8x8-v0
class MGDK8x8(ObsGrids7x7):
	pass

#MiniGrid-FourRooms-v0
class MGFR(ObsGrids7x7):
	pass



# Maze Like environments

#MiniGrid-SimpleCrossingS9N1-v0
class MGSCS9N1(ObsGrids7x7):
	pass

#MiniGrid-SimpleCrossingS9N2-v0
class MGSCS9N2(ObsGrids7x7):
	pass

#MiniGrid-SimpleCrossingS9N3-v0
class MGSCS9N3(ObsGrids7x7):
	pass

#MiniGrid-SimpleCrossingS11N5-v0
class MGSCS11N5(ObsGrids7x7):
	pass



# Lava Environments (Just the first 2 easier ones)

#MiniGrid-LavaCrossingS9N1-v0
class MGLCS9N1(ObsGrids7x7):
	pass

#MiniGrid-LavaCrossingS9N2-v0
class MGLCS9N2(ObsGrids7x7):
	pass



# Key corridor environment

#MiniGrid-KeyCorridorS3R1-v0
class MGKCS3R1(ObsGrids7x7):
	pass

#MiniGrid-KeyCorridorS3R2-v0
class MGKCS3R2(ObsGrids7x7):
	pass

#MiniGrid-KeyCorridorS3R3-v0
class MGKCS3R3(ObsGrids7x7):
	pass



# Obstructed maze environment

#MiniGrid-ObstructedMaze-1Dl-v0
class MGOM1Dl(ObsGrids7x7):
	pass

#MiniGrid-ObstructedMaze-1Dlh-v0
class MGOM1Dlh(ObsGrids7x7):
	pass

#MiniGrid-ObstructedMaze-1Dlhb-v0
class MGOM1Dlhb(ObsGrids7x7):
	pass


# #OLD STUFF NO LONGER REQUIRED, I WANTED TO USE CLASS INHERITANCE TO MAKE THINGS CLEANER

# #MiniGrid-Empty-8x8-v0
# class MGER8x8(data.Dataset):
# 	def __init__(self, data_folder):
# 		self.data_folder = data_folder
# 		self.training_file = 'training.pt'
# 		self.data = torch.load(os.path.join(self.data_folder, self.training_file))
# 		self.mean = torch.load(os.path.join(self.data_folder, 'mean.pt'))
# 		# self.std = torch.load(os.path.join(self.data_folder, 'std.pt'))*8
# 		self.max_vals = torch.load(os.path.join(self.data_folder, 'max_vals.pt'))*1.2
# 		self.transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(self.mean, self.max_vals)])
# 		# print (self.mean)
# 		# print (self.std)

# 	def __getitem__(self, index):
# 		data_item = self.data[index]
# 		# print (data_item.reshape(-1))
# 		# normalized_DI = torch.Tensor(data_item) / 255.0

# 		# for i in range(3):
# 		# 	normalized_DI[:,:,i] = (normalized_DI[:,:,i] - self.mean[i]) / self.std[i]
# 		# print(normalized_DI)

# 		# print (data_item.reshape(-1))
# 		# exit()
# 		# print (self.data.dtype)
# 		# exit()
# 		data_item = Image.fromarray(data_item)
# 		# print(list(data_item.getdata()))
# 		# exit()
# 		if self.transform is not None:
# 			data_item = self.transform(data_item)
# 		# print(list(data_item.getdata()))
# 		# exit()
# 		# print (data_item)
# 		# print (data_item.shape)
# 		# exit()
# 		return data_item.reshape(-1)

# 	def __len__(self):
# 		return len(self.data)


# #MiniGrid-DoorKey-8x8-v0
# class MGDK8x8(data.Dataset):
# 	def __init__(self, data_folder):
# 		self.data_folder = data_folder
# 		self.training_file = 'training.pt'
# 		self.data = torch.load(os.path.join(self.data_folder, self.training_file))
# 		self.mean = torch.load(os.path.join(self.data_folder, 'mean.pt'))
# 		# self.std = torch.load(os.path.join(self.data_folder, 'std.pt'))*8
# 		self.max_vals = torch.load(os.path.join(self.data_folder, 'max_vals.pt'))*1.2
# 		self.transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(self.mean, self.max_vals)])
# 		# print (self.mean)
# 		# print (self.std)

# 	def __getitem__(self, index):
# 		data_item = self.data[index]
# 		# print (data_item.reshape(-1))
# 		# normalized_DI = torch.Tensor(data_item) / 255.0

# 		# for i in range(3):
# 		# 	normalized_DI[:,:,i] = (normalized_DI[:,:,i] - self.mean[i]) / self.std[i]
# 		# print(normalized_DI)

# 		# print (data_item.reshape(-1))
# 		# exit()
# 		# print (self.data.dtype)
# 		# exit()
# 		data_item = Image.fromarray(data_item)
# 		# print(list(data_item.getdata()))
# 		# exit()
# 		if self.transform is not None:
# 			data_item = self.transform(data_item)
# 		# print(list(data_item.getdata()))
# 		# exit()
# 		# print (data_item)
# 		# print (data_item.shape)
# 		# exit()
# 		return data_item.reshape(-1)

# 	def __len__(self):
# 		return len(self.data)



# #MiniGrid-FourRooms-v0
# class MGFR(data.Dataset):
# 	def __init__(self, data_folder):
# 		self.data_folder = data_folder
# 		self.training_file = 'training.pt'
# 		self.data = torch.load(os.path.join(self.data_folder, self.training_file))
# 		self.mean = torch.load(os.path.join(self.data_folder, 'mean.pt'))
# 		# self.std = torch.load(os.path.join(self.data_folder, 'std.pt'))*8
# 		self.max_vals = torch.load(os.path.join(self.data_folder, 'max_vals.pt'))*1.2
# 		self.transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(self.mean, self.max_vals)])
# 		# print (self.mean)
# 		# print (self.std)

# 	def __getitem__(self, index):
# 		data_item = self.data[index]
# 		# print (data_item.reshape(-1))
# 		# normalized_DI = torch.Tensor(data_item) / 255.0

# 		# for i in range(3):
# 		# 	normalized_DI[:,:,i] = (normalized_DI[:,:,i] - self.mean[i]) / self.std[i]
# 		# print(normalized_DI)

# 		# print (data_item.reshape(-1))
# 		# exit()
# 		# print (self.data.dtype)
# 		# exit()
# 		data_item = Image.fromarray(data_item)
# 		# print(list(data_item.getdata()))
# 		# exit()
# 		if self.transform is not None:
# 			data_item = self.transform(data_item)
# 		# print(list(data_item.getdata()))
# 		# exit()
# 		# print (data_item)
# 		# print (data_item.shape)
# 		# exit()
# 		return data_item.reshape(-1)

# 	def __len__(self):
# 		return len(self.data)