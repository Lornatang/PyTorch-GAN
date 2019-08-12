# Copyright 2019 Lorna Authors. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 (the "License");
#   you may not use this file except in compliance with the License.
#   You may obtain a copy of the License at
#
#       http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

import argparse
import os

import torch
import numpy as np
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as vutils
from torch import nn
from torch import optim
from torch.backends import cudnn
from torch.utils import data
from torch.autograd import Variable

parser = argparse.ArgumentParser()
parser.add_argument("--n_epochs", type=int, default=200, help="number of epochs of training")
parser.add_argument("--batch_size", type=int, default=64, help="size of the batches")
parser.add_argument("--lr", type=float, default=0.0002, help="adam: learning rate")
parser.add_argument("--b1", type=float, default=0.5, help="adam: decay of first order momentum of gradient")
parser.add_argument("--b2", type=float, default=0.999, help="adam: decay of first order momentum of gradient")
parser.add_argument("--n_cpu", type=int, default=8, help="number of cpu threads to use during batch generation")
parser.add_argument("--latent_dim", type=int, default=100, help="dimensionality of the latent space")
parser.add_argument("--img_size", type=int, default=28, help="size of each image dimension")
parser.add_argument("--channels", type=int, default=1, help="number of image channels")
parser.add_argument("--sample_interval", type=int, default=400, help="interval betwen image samples")
parser.add_argument("--mode", type=str, default='train', help="train models or test model")
opt = parser.parse_args()
print(opt)

img_shape = (opt.channels, opt.img_size, opt.img_size)

os.makedirs("./images", exist_ok=True)

cudnn.benchmark = True

dataset = dset.MNIST(root="~/pytorch_datasets", download=True,
                     transform=transforms.Compose([
                         transforms.Resize(opt.img_size),
                         transforms.ToTensor(),
                         transforms.Normalize((0.5,), (0.5,))
                     ]))

assert dataset

dataloader = torch.utils.data.DataLoader(dataset,
                                         batch_size=opt.batch_size,
                                         shuffle=True)

# Define what device we are using
print("CUDA Available: ", torch.cuda.is_available())
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class Generator(nn.Module):
    """
        Generate models.
    """
    def __init__(self):
        super(Generator, self).__init__()

        def block(in_features, out_features, normalize=True):
            layers = [nn.Linear(in_features, out_features)]
            if normalize:
                layers.append(nn.BatchNorm1d(out_features, 0.8))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers

        self.model = nn.Sequential(
            *block(opt.latent_dim, 128, normalize=False),
            *block(128, 256),
            *block(256, 512),
            *block(512, 1024),
            nn.Linear(1024, int(np.prod(img_shape))),
            nn.Tanh()
        )

    def forward(self, z):
        """
            Forward propagation function
        Args:
            z: Noise under normal distribution
        Returns:
            size: 28 * 28 * 1, gray image.
        """
        img = self.model(z)
        img = img.view(img.size(0), *img_shape)
        return img


class Discriminator(nn.Module):
    """
        Discriminator models.
    """
    def __init__(self):
        super(Discriminator, self).__init__()

        self.model = nn.Sequential(
            nn.Linear(int(np.prod(img_shape)), 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(256, 1),
            nn.Sigmoid(),
        )

    def forward(self, img):
        """
            Forward propagation function
        Args:
            img: size: 28 * 28 * 1, gray image.
        Returns:
            size: real or fake label.
        """
        img_flat = img.view(img.size(0), -1)
        validity = self.model(img_flat)

        return validity


# Loss function
criterion = nn.BCELoss().to(device)

# Initialize generator and discriminator
netG = Generator().to(device)
netD = Discriminator().to(device)

# Optimizers
optimizer_G = optim.Adam(netG.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))
optimizer_D = optim.Adam(netD.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))

fixed_noise = torch.randn(opt.batch_size, opt.latent_dim, 1, 1, device=device)

# ----------
#  Training
# ----------
def train():
    for epoch in range(opt.n_epochs):
        for i, (data, _) in enumerate(dataloader, 0):
            ###############################################
            # Step 1: Train Discriminator network
            # math: `maximize log(D(x)) + log(1 - D(G(z)))`
            ###############################################

            # load train datasets
            netD.zero_grad()
            real_data = data[0].to(device)
            batch_size = real_data.size(0)

            # real img label is 1, fake img label is 0.
            real_label = torch.full((batch_size,), 1, device=device)
            fake_label = torch.full((batch_size,), 0, device=device)
            # sample noise as generator input
            noise = torch.randn(batch_size, opt.latent_dim, 1, 1, device=device)

            # train with real datasets
            output = netD(real_data)
            errD_real = criterion(output, real_label)
            errD_real.backward()
            D_x = output.mean().item()

            # train with fake datasets
            fake = netG(noise)
            output = netD(fake.detach())
            errD_fake = criterion(output, fake_label)
            errD_fake.backward()
            D_G_z1 = output.mean().item()
            errD = errD_real + errD_fake
            optimizer_D.step()

            ###############################################
            # Step 2: Train Generator network
            # math: `maximize log(D(G(z)))`
            ###############################################
            netG.zero_grad()
            output = netD(fake)
            errG = criterion(output, real_label)
            errG.backward()
            D_G_z2 = output.mean().item()
            optimizer_G.step()

            print(f"Epoch->[{epoch}/{opt.niter}] "
                  f"batch->[{i}/{len(dataloader)}] "
                  f"Loss_D: {errD.item():.4f} "
                  f"Loss_G: {errG.item():.4f} "
                  f"D(x): {D_x.item():.4f} "
                  f"D(G(z)): {D_G_z1.item():.4f}/{D_G_z2.item():.4f}")
            if i % 100 == 0:
                vutils.save_image(real_data,
                                  f"{opt.outf}/real_samples.png",
                                  normalize=True)
                fake = netG(fixed_noise)
                vutils.save_image(fake.detach(),
                                  f"{opt.outf}/fake_samples_epoch_{epoch:03d}.png",
                                  normalize=True)

        # do checkpointing
        torch.save(netG, f"{opt.outf}/netG_epoch_{epoch}.pth")
        torch.save(netD, f"{opt.outf}/netD_epoch_{epoch}.pth")


# ----------
#  Testing
# ----------
def test():
    netG = torch.load(f"{opt.outf}/netG_epoch_{opt.niter}.pth")
    fake = netG(fixed_noise)
    vutils.save_image(fake.detach(),
                      f"{opt.result}/fake_samples.png",
                      normalize=True)


if __name__ == '__main__':
    if opt.mode == 'train':
        train()
    elif opt.mode == 'test':
        test()
    else:
        print(opt)
