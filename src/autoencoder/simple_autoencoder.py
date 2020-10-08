# __author__ = 'SherlockLiao'

import os
import time
import argparse

import torch
import torchvision
from torch import nn
# from torch.autograd import Variable
from torch.utils.data import DataLoader
# from torchvision import transforms
from torchvision.utils import save_image

# if not os.path.exists('mlp_img'):
#     os.mkdir('mlp_img')

def unNormalize(data, mean, scale_normalizer, view_row=3, view_col=3):
    # print(mean, std)
    # scale_normalizer can be a multiple of the std dev or the maximum channel value
    uN_data = data.clone().reshape(-1, 3, view_row, view_col)
    for i in range(3):
        uN_data[:, i, :, :] = (uN_data[:, i, :, :]*scale_normalizer[i] + mean[i])*255.0
    return uN_data


class autoencoder(nn.Module):
    def __init__(self, big_obs_grid=False):
        super(autoencoder, self).__init__()
        if not big_obs_grid:
            #For 3x3 observation grid
            self.latent_space_size = 16
            self.encoder = nn.Sequential(
                nn.Linear(27, 25),
                nn.ReLU(),
                nn.Linear(25, self.latent_space_size))
            self.decoder = nn.Sequential(
                nn.Linear(self.latent_space_size, 25),
                nn.ReLU(),
                nn.Linear(25, 27),
                nn.Tanh())
        else:
            #For 7x7 observation grid
            self.latent_space_size = 64
            self.encoder = nn.Sequential(
                nn.Linear(147, 96),
                nn.ReLU(),
                nn.Linear(96, self.latent_space_size))
            self.decoder = nn.Sequential(
                nn.Linear(self.latent_space_size, 96),
                nn.ReLU(),
                nn.Linear(96, 147),
                nn.Tanh())

    def forward(self, x, getLatent=False):
        x = self.encoder(x)
        # print (x.shape)
        if not getLatent:
            x = self.decoder(x)
        return x

if __name__ == "__main__":
    from minigrid_datasets import MGER6x6, MGMRN2S4, MGDK6x6, MGER8x8, MGDK8x8, MGFR
    from minigrid_datasets import MGSCS9N1, MGSCS9N2, MGSCS9N3, MGSCS11N5, MGLCS9N1, MGLCS9N2
    from minigrid_datasets import MGKCS3R1, MGKCS3R2, MGKCS3R3, MGOM1Dl, MGOM1Dlh, MGOM1Dlhb

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
        default=256
    )
    parser.add_argument(
        "--num_epochs",
        type=int,
        help="Number of Epochs to run over dataset",
        default=100
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        help="Learning Rate of the Autoencoder",
        default=1e-4
    )
    parser.add_argument(
        "--starting_model_path",
        help="path for model to start with"
    )
    parser.add_argument(
        "--save_folder",
        help="folder to save data in",
        default='models/MGER6x6/ExpName_LatentSize16_withLargeStd_withLowLR_WD'
    )
    parser.add_argument(
        "--use_gpu",
        action="store_true",
        help="GPU selection",
        default=False
    )

    args = parser.parse_args()
    
    use_cuda = torch.cuda.is_available() and args.use_gpu
    device = torch.device("cuda" if use_cuda else "cpu")
    
    if not os.path.exists(args.save_folder):
        os.makedirs(args.save_folder)

    num_epochs = args.num_epochs
    batch_size = args.batch_size
    learning_rate = args.learning_rate

    big_obs_grid = False
    if args.dataset_name == 'MGER6x6':
        dataset = MGER6x6(args.dataset_folder)
    elif args.dataset_name == 'MGMRN2S4':
        dataset = MGMRN2S4(args.dataset_folder)
    elif args.dataset_name == 'MGDK6x6':
        dataset = MGDK6x6(args.dataset_folder)
    else:
        big_obs_grid = True
        # Initially tried environments
        if args.dataset_name == 'MGER8x8':
            dataset = MGER8x8(args.dataset_folder)
        elif args.dataset_name == 'MGDK8x8':
            dataset = MGDK8x8(args.dataset_folder)
        elif args.dataset_name == 'MGFR':
            dataset = MGFR(args.dataset_folder)
        # Maze Like environments
        elif args.dataset_name == 'MGSCS9N1':
            dataset = MGSCS9N1(args.dataset_folder)
        elif args.dataset_name == 'MGSCS9N2':
            dataset = MGSCS9N2(args.dataset_folder)
        elif args.dataset_name == 'MGSCS9N3':
            dataset = MGSCS9N3(args.dataset_folder)
        elif args.dataset_name == 'MGSCS11N5':
            dataset = MGSCS11N5(args.dataset_folder)
        # Lava Environments (Just the first 2 easier ones)
        elif args.dataset_name == 'MGLCS9N1':
            dataset = MGLCS9N1(args.dataset_folder)
        elif args.dataset_name == 'MGLCS9N2':
            dataset = MGLCS9N2(args.dataset_folder)
        # Key corridor environment
        elif args.dataset_name == 'MGKCS3R1':
            dataset = MGKCS3R1(args.dataset_folder)
        elif args.dataset_name == 'MGKCS3R2':
            dataset = MGKCS3R2(args.dataset_folder)
        elif args.dataset_name == 'MGKCS3R3':
            dataset = MGKCS3R3(args.dataset_folder)
        # Obstructed maze environment
        elif args.dataset_name == 'MGOM1Dl':
            dataset = MGOM1Dl(args.dataset_folder)
        elif args.dataset_name == 'MGOM1Dlh':
            dataset = MGOM1Dlh(args.dataset_folder)
        elif args.dataset_name == 'MGOM1Dlhb':
            dataset = MGOM1Dlhb(args.dataset_folder)

    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    model = autoencoder(big_obs_grid).to(device)
    if args.starting_model_path is not None:
        model.load_state_dict(torch.load(args.starting_model_path))
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-6)

    start_time = time.time()
    for epoch in range(num_epochs):
        # print ('Now Running Epoch: [{}/{}]'.format(epoch, num_epochs))
        losses = []
        for data in dataloader:
            data = data.to(device)
    #         exit()
            # print (data)
            # print (data.shape)
    #         exit()
    #         print (output[0,:])
    #         output1 = model(data[0, :])
    #         print (output1)
            # unNormalized_data = unNormalize(data, dataset.mean, dataset.std)
            # print('---------------------')
            # for i in unNormalized_data:
            #     print (i)
            # exit()
    #         img = data
    #         img = img.view(img.size(0), -1)
    #         # img = Variable(img).cuda()
    #         # ===================forward=====================
    #         output = model(img)
            output = model(data)
            # print (output)
            # print (output.shape)
            loss = criterion(output, data)
            # ===================backward====================
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            losses.append(loss.detach().item())
        # ===================log========================
        print('Epoch [{}/{}], Loss:{:.10f}'.format(epoch + 1, num_epochs, torch.Tensor(losses).mean().item()))
        if epoch % 5 == 0:
            # pic = to_img(output.cpu().data)
            # save_image(pic, './mlp_img/image_{}.png'.format(epoch))
            torch.save(model.state_dict(), os.path.join(args.save_folder, 'autoencoder_{}.pth'.format(epoch)))
    
    total_time = time.time() - start_time
    print ('Total Time Taken: ', total_time)
    
    torch.save(model.state_dict(), os.path.join(args.save_folder, 'autoencoder_final.pth'))