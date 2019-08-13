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

"""Generative Adversarial Networks (GANs) are one of the most interesting ideas
in computer science today. Two models are trained simultaneously by
an adversarial process. A generator ("the artist") learns to create images
that look real, while a discriminator ("the art critic") learns
to tell real images apart from fakes.
"""

import argparse
import os
import random
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as vutils

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', required=True, help='mnist | fashion-mnist')
parser.add_argument('--dataroot', required=True, help='path to dataset')
parser.add_argument('--workers', type=int, help='number of data loading workers', default=2)
parser.add_argument('--batch_size', type=int, default=256, help='inputs batch size')
parser.add_argument('--image_size', type=int, default=28, help='the height / width of the inputs image to network')
parser.add_argument('--nz', type=int, default=100, help='size of the latent z vector')
parser.add_argument('--n_epochs', type=int, default=25, help='number of epochs to train for')
parser.add_argument('--lr', type=float, default=0.001, help='learning rate, default=0.0002')
parser.add_argument('--beta1', type=float, default=0.5, help='beta1 for adam. default=0.5')
parser.add_argument('--beta2', type=float, default=0.999, help='beta2 for adam. default=0.999')
parser.add_argument('--cuda', action='store_true', help='enables cuda')
parser.add_argument('--ngpu', type=int, default=1, help='number of GPUs to use')
parser.add_argument('--netG', default='', help="path to netG (to continue training)")
parser.add_argument('--netD', default='', help="path to netD (to continue training)")
parser.add_argument('--outf', default='.', help='folder to output images and model checkpoints')
parser.add_argument('--manualSeed', type=int, help='manual seed')

opt = parser.parse_args()
print(opt)

try:
    os.makedirs(opt.outf)
except OSError:
    pass

if opt.manualSeed is None:
    opt.manualSeed = random.randint(1, 10000)
print("Random Seed: ", opt.manualSeed)
random.seed(opt.manualSeed)
torch.manual_seed(opt.manualSeed)

cudnn.benchmark = True

if torch.cuda.is_available() and not opt.cuda:
    print("WARNING: You have a CUDA device, so you should probably run with --cuda")

if opt.dataset == 'fashion-mnist':
    dataset = dset.FashionMNIST(root=opt.dataroot, download=True,
                                transform=transforms.Compose([
                                    transforms.Resize(opt.image_size),
                                    transforms.ToTensor(),
                                    transforms.Normalize((0.5,), (0.5,)),
                                ]))
    nc = 1
elif opt.dataset == 'mnist':
    dataset = dset.MNIST(root=opt.dataroot, download=True,
                         transform=transforms.Compose([
                             transforms.Resize(opt.image_size),
                             transforms.ToTensor(),
                             transforms.Normalize([0.5], [0.5]),
                         ]))
    nc = 1

assert dataset
dataloader = torch.utils.data.DataLoader(dataset, batch_size=opt.batch_size,
                                         shuffle=True, num_workers=int(opt.workers))

device = torch.device("cuda:0" if opt.cuda else "cpu")
ngpu = int(opt.ngpu)
nz = int(opt.nz)


# custom weights initialization called on netG and netD
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)


class Generator(nn.Module):
    def __init__(self, gpus):
        super(Generator, self).__init__()
        self.ngpu = gpus
        self.main = nn.Sequential(
            nn.Linear(nz, 1024),
            nn.ReLU(inplace=True),
            nn.Linear(1024, 1024),
            nn.ReLU(inplace=True),
            nn.Linear(1024, 784),
            nn.Tanh()
        )

    def forward(self, inputs):
        if inputs.is_cuda and self.ngpu > 1:
            outputs = nn.parallel.data_parallel(self.main, inputs, range(self.ngpu))
        else:
            outputs = self.main(inputs)
        return outputs


netG = Generator(ngpu).to(device)
netG.apply(weights_init)
if opt.netG != '':
    torch.load(opt.netG)
print(netG)


class Discriminator(nn.Module):
    def __init__(self, gpus):
        super(Discriminator, self).__init__()
        self.ngpu = gpus
        self.main = nn.Sequential(
            nn.Linear(784, 256),
            nn.LeakyReLU(0.01, inplace=True),
            nn.Linear(256, 256),
            nn.LeakyReLU(0.01, inplace=True),
            nn.Linear(256, 1),
        )

    def forward(self, inputs):
        if inputs.is_cuda and self.ngpu > 1:
            outputs = nn.parallel.data_parallel(self.main, inputs, range(self.ngpu))
        else:
            outputs = self.main(inputs)

        return outputs


netD = Discriminator(ngpu).to(device)
netD.apply(weights_init)
if opt.netD != '':
    torch.load(opt.netD)
print(netD)

criterion = nn.BCEWithLogitsLoss()

fixed_noise = torch.randn(opt.batch_size, nz, 1, 1, device=device)

# setup optimizer
optimizerD = optim.Adam(netD.parameters(), lr=opt.lr, betas=(opt.beta1, opt.beta2))
optimizerG = optim.Adam(netG.parameters(), lr=opt.lr, betas=(opt.beta1, opt.beta2))


def train():
    """ train model
    """
    for epoch in range(opt.n_epochs):
        for i, data in enumerate(dataloader, 0):
            ############################
            # (1) Update D network: maximize log(D(x)) + log(1 - D(G(z)))
            ###########################
            # train with real
            netD.zero_grad()
            real_data = data[0].to(device)
            batch_size = real_data.size(0)

            # real data label is 1, fake data label is 0.
            real_label = torch.full((batch_size,), 1, device=device)
            fake_label = torch.full((batch_size,), 0, device=device)

            output = netD(real_data)
            errD_real = criterion(output, real_label)
            errD_real.backward()
            D_x = output.mean().item()

            # train with fake
            noise = torch.randn(batch_size, nz, 1, 1, device=device)
            fake = netG(noise)
            output = netD(fake.detach())
            errD_fake = criterion(output, fake_label)
            errD_fake.backward()
            D_G_z1 = output.mean().item()
            errD = errD_real + errD_fake
            optimizerD.step()

            ############################
            # (2) Update G network: maximize log(D(G(z)))
            ###########################
            netG.zero_grad()
            output = netD(fake)
            errG = criterion(output, real_label)
            errG.backward()
            D_G_z2 = output.mean().item()
            optimizerG.step()

            print(f"Epoch->[{epoch}/{opt.niter}] "
                  f"Progress->[{i / len(dataloader) * 100}]% "
                  f"Loss_D: {errD.item():.4f} "
                  f"Loss_G: {errG.item():.4f} "
                  f"D(x): {D_x:.4f} "
                  f"D(G(z)): {D_G_z1:.4f} / {D_G_z2:.4f}")

            if i % 100 == 0:
                vutils.save_image(real_data, f"{opt.outf}/real_samples.png", normalize=True)
                fake = netG(fixed_noise)
                vutils.save_image(fake.detach(), f"{opt.outf}/fake_samples_epoch_{epoch:03d}.png", normalize=True)

        # do checkpointing
        torch.save(netG, f"{opt.outf}/netG_epoch_{epoch:03d}")
        torch.save(netD, f"{opt.outf}/netD_epoch_{epoch:03d}")


if __name__ == '__main__':
    train()