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

import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.optim as optim
import torch.utils.data
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as vutils

parser = argparse.ArgumentParser()
parser.add_argument('--dataroot', type=str, default='~/pytorch_datasets', help='path to dataset')
parser.add_argument('--workers', type=int, help='number of data loading workers', default=8)
parser.add_argument('--batch_size', type=int, default=256, help='inputs batch size')
parser.add_argument('--image_size', type=int, default=28, help='the height / width of the inputs image to network')
parser.add_argument('--nz', type=int, default=100, help='size of the latent z vector')
parser.add_argument('--n_epochs', type=int, default=50, help='number of epochs to train for')
parser.add_argument('--lr', type=float, default=0.0002, help='learning rate, default=0.0002')
parser.add_argument('--beta1', type=float, default=0.5, help='beta1 for adam. default=0.5')
parser.add_argument('--beta2', type=float, default=0.999, help='beta2 for adam. default=0.999')
parser.add_argument('--cuda', action='store_true', help='enables cuda')
parser.add_argument('--ngpu', type=int, default=1, help='number of GPUs to use')
parser.add_argument('--netG', default='', help="path to netG (to continue training)")
parser.add_argument('--netD', default='', help="path to netD (to continue training)")
parser.add_argument('--out_images', default='./imgs', help='folder to output images')
parser.add_argument('--out_folder', default='./checkpoints', help='folder to output model checkpoints')
parser.add_argument('--manualSeed', type=int, help='manual seed')
parser.add_argument('--phase', type=str, default='train', help='model mode. default=`train`')

opt = parser.parse_args()
print(opt)

try:
  os.makedirs(opt.out_images)
  os.makedirs(opt.out_folder)
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

device = torch.device("cuda:0" if opt.cuda else "cpu")

ngpu = int(opt.ngpu)
nz = int(opt.nz)


class Generator(nn.Module):
  """ generate model
  """
  def __init__(self, ngpu):
    super(Generator, self).__init__()
    self.ngpu = ngpu

    self.main = nn.Sequential(
      nn.Linear(nz, 256),
      nn.Linear(256, 512),
      nn.Linear(512, 1024),
      nn.Linear(1024, 784),
      nn.Tanh()
    )

  def forward(self, inputs):
    """ forward layer
    Args:
      inputs: input tensor data.
    Returns:
      forwarded data.
    """
    if inputs.is_cuda and self.ngpu > 1:
      outputs = nn.parallel.data_parallel(self.main, inputs, range(self.ngpu))
    else:
      outputs = self.main(inputs)
    return outputs.view(outputs.size(0), *(1, 28, 28))


class Discriminator(nn.Module):
  """ discriminate model
  """
  def __init__(self, ngpu):
    super(Discriminator, self).__init__()
    self.ngpu = ngpu

    self.main = nn.Sequential(
      nn.Linear(784, 512),
      nn.Linear(512, 256),
      nn.Linear(256, 1),
      nn.Sigmoid()
    )

  def forward(self, inputs):
    """ forward layer
    Args:
      inputs: input tensor data.
    Returns:
      forwarded data.
    """
    if inputs.is_cuda and self.ngpu > 1:
      outputs = nn.parallel.data_parallel(self.main, inputs, range(self.ngpu))
    else:
      inputs = inputs.view(inputs.size(0), -1)
      outputs = self.main(inputs)
    return outputs


fixed_noise = torch.randn(opt.batch_size, nz, device=device)


def train():
  """ train model
  """
  ################################################
  #               load train dataset
  ################################################
  dataset = dset.MNIST(root=opt.dataroot,
                       download=True,
                       transform=transforms.Compose([
                         transforms.Resize(opt.image_size),
                         transforms.ToTensor(),
                         transforms.Normalize((0.5,), (0.5,)),
                       ]))

  assert dataset
  dataloader = torch.utils.data.DataLoader(dataset, batch_size=opt.batch_size,
                                           shuffle=True, num_workers=int(opt.workers))

  ################################################
  #               load model
  ################################################
  netG = Generator(ngpu).to(device)
  if opt.netG != '':
    netG.load_state_dict(torch.load(opt.netG, map_location=lambda storage, loc: storage))
  print(netG)

  netD = Discriminator(ngpu).to(device)
  if opt.netD != '':
    netD.load_state_dict(torch.load(opt.netD, map_location=lambda storage, loc: storage))
  print(netD)

  ################################################
  #           Binary Cross Entropy
  ################################################
  criterion = nn.BCELoss()

  ################################################
  #            Use Adam optimizer
  ################################################
  optimizerD = optim.Adam(netD.parameters(), lr=opt.lr, betas=(opt.beta1, opt.beta2))
  optimizerG = optim.Adam(netG.parameters(), lr=opt.lr, betas=(opt.beta1, opt.beta2))

  ################################################
  #               print args
  ################################################
  print("########################################")
  print(f"train dataset path: {opt.dataroot}")
  print(f"work thread: {opt.workers}")
  print(f"batch size: {opt.batch_size}")
  print(f"image size: {opt.image_size}")
  print(f"Epochs: {opt.n_epochs}")
  print(f"Noise size: {opt.nz}")
  print("########################################")
  print(f"loading pretrain model successful!\n")
  print("Starting trainning!")
  for epoch in range(opt.n_epochs):
    for i, data in enumerate(dataloader):
      ##############################################
      # (1) Update D network: maximize log(D(x)) + log(1 - D(G(z)))
      ##############################################
      # train with real
      netD.zero_grad()
      real_data = data[0].to(device)
      batch_size = real_data.size(0)

      # real data label is 1, fake data label is 0.
      real_label = torch.full((batch_size,), 1, device=device)
      fake_label = torch.full((batch_size,), 0, device=device)

      # real_data = data.view(100, -1)
      output = netD(real_data).view(-1)
      errD_real = criterion(output, real_label)
      errD_real.backward()
      D_x = output.mean().item()

      # train with fake
      noise = torch.randn(batch_size, nz, device=device)
      fake = netG(noise)
      output = netD(fake.detach()).view(-1)
      errD_fake = criterion(output, fake_label)
      errD_fake.backward()
      D_G_z1 = output.mean().item()
      errD = errD_real + errD_fake
      optimizerD.step()

      ##############################################
      # (2) Update G network: maximize log(D(G(z)))
      ##############################################
      netG.zero_grad()
      output = netD(fake).view(-1)
      errG = criterion(output, real_label)
      errG.backward()
      D_G_z2 = output.mean().item()
      optimizerG.step()
      print(f"Epoch->[{epoch + 1:03d}/{opt.n_epochs:03d}] "
            f"Progress->{i / len(dataloader) * 100:4.2f}% "
            f"Loss_D: {errD.item():.4f} "
            f"Loss_G: {errG.item():.4f} "
            f"D(x): {D_x:.4f} "
            f"D(G(z)): {D_G_z1:.4f} / {D_G_z2:.4f}", end="\r")

      if i % 100 == 0:
        vutils.save_image(real_data, f"{opt.out_images}/real_samples.png", normalize=True)
        with torch.no_grad():
          fake = netG(fixed_noise).detach().cpu()
        vutils.save_image(fake, f"{opt.out_images}/fake_samples_epoch_{epoch + 1:03d}.png", normalize=True)

    # do checkpointing
    torch.save(netG.state_dict(), f"{opt.out_folder}/netG_epoch_{epoch + 1:03d}.pth")
    torch.save(netD.state_dict(), f"{opt.out_folder}/netD_epoch_{epoch + 1:03d}.pth")


def generate():
  """ random generate fake image.
  """
  ################################################
  #               load model
  ################################################
  print(f"Load model...\n")
  netG = Generator(ngpu).to(device)
  netG.load_state_dict(torch.load(opt.netG, map_location=lambda storage, loc: storage))
  print(f"Load model successful!")
  one_noise = torch.randn(1, nz, device=device)
  with torch.no_grad():
    fake = netG(one_noise).detach().cpu()
  vutils.save_image(fake, f"{opt.out_images}/fake.png", normalize=True)


if __name__ == '__main__':
  if opt.phase == 'train':
    train()
  elif opt.phase == 'generate':
    generate()
  else:
    print(opt)
