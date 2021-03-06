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
import glob
import os
import random

import IPython
import imageio
import torch
import torch.utils.data.dataloader
from IPython import display
from torch import nn
from torch import optim
from torch.backends import cudnn as cudnn
from torchvision import datasets as dset
from torchvision import transforms
from torchvision import utils as vutils

parser = argparse.ArgumentParser()
parser.add_argument('--dataroot', type=str, default='~/pytorch_datasets', help='path to dataset')
parser.add_argument('--workers', type=int, help='number of data loading workers', default=8)
parser.add_argument('--batch_size', type=int, default=64, help='inputs batch size')
parser.add_argument('--image_size', type=int, default=28, help='the height / width of the inputs image to network')
parser.add_argument('--nz', type=int, default=100, help='size of the latent z vector')
parser.add_argument('--n_epochs', type=int, default=200, help='number of epochs to train for')
parser.add_argument('--lr', type=float, default=0.0002, help='learning rate, default=0.0002')
parser.add_argument('--beta1', type=float, default=0.5, help='beta1 for adam. default=0.5')
parser.add_argument('--beta2', type=float, default=0.999, help='beta2 for adam. default=0.999')
parser.add_argument('--cuda', action='store_true', help='enables cuda')
parser.add_argument('--ngpu', type=int, default=1, help='number of GPUs to use')
parser.add_argument('--netG', default='', help="path to netG (to continue training)")
parser.add_argument('--netD', default='', help="path to netD (to continue training)")
parser.add_argument('--out_images', default='./imgs', help='folder to output images')
parser.add_argument('--checkpoints_dir', default='./checkpoints', help='folder to output model checkpoints')
parser.add_argument('--manualSeed', type=int, help='manual seed')
parser.add_argument('--phase', type=str, default='train', help='model mode. default=`train`')
parser.add_argument('--sample_size', type=int, default=1000, help='generate 1000 pic use classifier.')

opt = parser.parse_args()

try:
  os.makedirs(opt.out_images)
  os.makedirs("./unknown")
except OSError:
  pass

if opt.manualSeed is None:
  opt.manualSeed = random.randint(1, 10000)
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

    def block(in_features, out_features, normalize=True):
      """ simple layer struct.
      Args:
        in_features: input feature.
        out_features: output feature.
        normalize: is normalize.
      Returns:
        new layer.
      """
      layers = [nn.Linear(in_features, out_features)]
      if normalize:
        layers.append(nn.BatchNorm1d(out_features, 0.8))
      layers.append(nn.LeakyReLU(0.2, inplace=True))
      return layers

    self.main = nn.Sequential(
      *block(nz, 128, normalize=False),
      *block(128, 256),
      *block(256, 512),
      *block(512, 1024),
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
      nn.LeakyReLU(0.2, inplace=True),
      nn.Linear(512, 256),
      nn.LeakyReLU(0.2, inplace=True),
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
    inputs = inputs.view(inputs.size(0), -1)
    if inputs.is_cuda and self.ngpu > 1:
      outputs = nn.parallel.data_parallel(self.main, inputs, range(self.ngpu))
    else:
      outputs = self.main(inputs)
    return outputs


fixed_noise = torch.randn(opt.batch_size, nz, device=device)


def train():
  """ train model
  """
  try:
    os.makedirs(opt.checkpoints_dir)
  except OSError:
    pass
  ################################################
  #               load train dataset
  ################################################
  dataset = dset.MNIST(root=opt.dataroot,
                       download=True,
                       transform=transforms.Compose([
                         transforms.Resize(opt.image_size),
                         transforms.ToTensor(),
                         transforms.Normalize([0.5], [0.5]),
                       ]))

  assert dataset
  dataloader = torch.utils.data.DataLoader(dataset, batch_size=opt.batch_size,
                                           shuffle=True, num_workers=int(opt.workers))

  ################################################
  #               load model
  ################################################
  if torch.cuda.device_count() > 1:
    netG = torch.nn.DataParallel(Generator(ngpu))
  else:
    netG = Generator(ngpu)
  if opt.netG != '':
    netG.load_state_dict(torch.load(opt.netG, map_location=lambda storage, loc: storage))

  if torch.cuda.device_count() > 1:
    netD = torch.nn.DataParallel(Discriminator(ngpu))
  else:
    netD = Discriminator(ngpu)
  if opt.netD != '':
    netD.load_state_dict(torch.load(opt.netD, map_location=lambda storage, loc: storage))

  netG.train()
  netG = netG.to(device)
  netD.train()
  netD = netD.to(device)
  print(netG)
  print(netD)

  ################################################
  #           Binary Cross Entropy
  ################################################
  criterion = nn.BCELoss()

  ################################################
  #            Use Adam optimizer
  ################################################
  optimizer_D = optim.Adam(netD.parameters(), lr=opt.lr, betas=(opt.beta1, opt.beta2))
  optimizer_G = optim.Adam(netG.parameters(), lr=opt.lr, betas=(opt.beta1, opt.beta2))

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
  print("Starting training!")
  for epoch in range(opt.n_epochs):
    for i, data in enumerate(dataloader):
      # get batch size data
      real_imgs = data[0].to(device)
      batch_size = real_imgs.size(0)

      # real data label is 1, fake data label is 0.
      valid = torch.full((batch_size,), 1, device=device)
      fake = torch.full((batch_size,), 0, device=device)
      # Sample noise as generator input
      z = torch.randn(batch_size, nz, device=device)

      ##############################################
      # (1) Update G network: maximize log(D(G(z)))
      ##############################################

      optimizer_G.zero_grad()

      # Generate a batch of images
      gen_imgs = netG(z)

      # Loss measures generator's ability to fool the discriminator
      fake_output = netD(gen_imgs)
      loss_G = criterion(fake_output, valid)

      loss_G.backward()
      optimizer_G.step()

      ##############################################
      # (2) Update D network: maximize log(D(x)) + log(1 - D(G(z)))
      ##############################################

      optimizer_D.zero_grad()

      # Measure discriminator's ability to classify real from generated samples
      real_output = netD(real_imgs)
      fake_output = netD(gen_imgs.detach())

      real_loss = criterion(real_output, valid)
      fake_loss = criterion(fake_output, fake)
      loss_D = (real_loss + fake_loss) / 2

      loss_D.backward()
      optimizer_D.step()

      print(f"Epoch->[{epoch + 1:03d}/{opt.n_epochs:03d}] "
            f"Progress->{i / len(dataloader) * 100:4.2f}% "
            f"Loss_D: {loss_D.item():.4f} "
            f"Loss_G: {loss_G.item():.4f}", end="\r")

      if i % 400 == 0:
        vutils.save_image(real_imgs, f"{opt.out_images}/real_samples.png", normalize=True)
        with torch.no_grad():
          generate_image = netG(fixed_noise).detach().cpu()
        vutils.save_image(generate_image, f"{opt.out_images}/fake_samples_epoch_{epoch + 1:03d}.png", normalize=True)

    # save model
    torch.save(netG.state_dict(), f"{opt.checkpoints_dir}/netG_epoch_{epoch + 1:03d}.pth")
    torch.save(netD.state_dict(), f"{opt.checkpoints_dir}/netD_epoch_{epoch + 1:03d}.pth")


def generate():
  """ random generate fake image.
  """
  ################################################
  #               load model
  ################################################
  print(f"Load model...\n")
  if torch.cuda.device_count() > 1:
    netG = torch.nn.DataParallel(Generator(ngpu))
  else:
    netG = Generator(ngpu)
  netG.to(device)
  netG.load_state_dict(torch.load(opt.netG, map_location=lambda storage, loc: storage))
  netG.eval()
  print(f"Load model successful!")
  with torch.no_grad():
    for i in range(opt.sample_size):
      z = torch.randn(1, nz, device=device)
      fake = netG(z).detach().cpu()
      vutils.save_image(fake, f"unknown/fake_{i:04d}.png", normalize=True)
  print(f"1000 images have been generated!")


def create_gif(file_name):
  """ auto generate gif
  Args:
      file_name:
  """
  with imageio.get_writer(file_name, mode='I') as writer:
    filenames = glob.glob(opt.out_images + '/' + '*.png')
    filenames = sorted(filenames)
    last = -1
    for i, filename in enumerate(filenames):
      frame = 2 * (i ** 0.5)
      if round(frame) > round(last):
        last = frame
      else:
        continue
      image = imageio.imread(filename)
      writer.append_data(image)
    image = imageio.imread(filename)
    writer.append_data(image)

  if IPython.version_info > (6, 2, 0, ''):
    display.Image(filename=file_name)


if __name__ == '__main__':
  if opt.phase == 'train':
    train()
    create_gif("gan.gif")
  elif opt.phase == 'generate':
    generate()
  else:
    print(opt)
