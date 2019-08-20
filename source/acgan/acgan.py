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
import numpy as np
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.optim as optim
import torch.utils.data
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as vutils
from IPython import display

parser = argparse.ArgumentParser()
parser.add_argument('--dataroot', type=str, default='~/pytorch_datasets', help='path to dataset')
parser.add_argument('--workers', type=int, help='number of data loading workers', default=8)
parser.add_argument('--batch_size', type=int, default=64, help='inputs batch size')
parser.add_argument('--image_size', type=int, default=32, help='the height / width of the inputs image to network')
parser.add_argument('--n_classes', type=int, default=10, help='number of classes for dataset')
parser.add_argument('--channels', type=int, default=1, help='number of image channels')
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

opt = parser.parse_args()

try:
  os.makedirs(opt.out_images)
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


def weights_init_normal(m):
  """ custom init weights
  Args:
    m: layer name.
  """
  classname = m.__class__.__name__
  if classname.find("Conv") != -1:
    torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
  elif classname.find("BatchNorm2d") != -1:
    torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
    torch.nn.init.constant_(m.bias.data, 0.0)


class Generator(nn.Module):
  """ generate model
  """

  def __init__(self, ngpu):
    super(Generator, self).__init__()
    self.ngpu = ngpu

    self.label_emb = nn.Embedding(opt.n_classes, nz)

    self.init_size = opt.image_size // 4  # Initial size before upsampling

    self.l1 = nn.Sequential(nn.Linear(nz, 128 * self.init_size ** 2))

    self.conv_blocks = nn.Sequential(
      nn.BatchNorm2d(128),
      nn.Upsample(scale_factor=2),
      nn.Conv2d(128, 128,
                kernel_size=(3, 3),
                stride=1,
                padding=1),
      nn.BatchNorm2d(128, 0.8),
      nn.LeakyReLU(0.2, inplace=True),
      nn.Upsample(scale_factor=2),
      nn.Conv2d(128, 64,
                kernel_size=(3, 3),
                stride=1,
                padding=1),
      nn.BatchNorm2d(64, 0.8),
      nn.LeakyReLU(0.2, inplace=True),
      nn.Conv2d(64, opt.channels,
                kernel_size=(3, 3),
                stride=1,
                padding=1),
      nn.Tanh()
    )

  def forward(self, inputs, labels):
    """ forward layer
    Args:
      inputs: input tensor data.
      labels: data label.
    Returns:
      forwarded data.
    """
    inputs = torch.mul(self.label_emb(labels), inputs)
    inputs = self.l1(inputs)
    inputs = inputs.view(inputs.shape[0], 128, self.init_size, self.init_size)
    if inputs.is_cuda and self.ngpu > 1:
      outputs = nn.parallel.data_parallel(self.conv_blocks, inputs, range(self.ngpu))
    else:
      outputs = self.conv_blocks(inputs)
    return outputs


class Discriminator(nn.Module):
  """ discriminate model
  """

  def __init__(self, ngpu):
    super(Discriminator, self).__init__()
    self.ngpu = ngpu

    def discriminator_block(in_filters, out_filters, bn=True):
      """ Returns layers of each discriminator block"""
      block = [nn.Conv2d(in_filters, out_filters, 3, 2, 1), nn.LeakyReLU(0.2, inplace=True), nn.Dropout2d(0.25)]
      if bn:
        block.append(nn.BatchNorm2d(out_filters, 0.8))
      return block

    self.conv_blocks = nn.Sequential(
      *discriminator_block(opt.channels, 16, bn=False),
      *discriminator_block(16, 32),
      *discriminator_block(32, 64),
      *discriminator_block(64, 128),
    )

    # The height and width of downsampled image
    ds_size = opt.image_size // 2 ** 4

    # Output layers
    self.adv_layer = nn.Sequential(nn.Linear(128 * ds_size ** 2, 1), nn.Sigmoid())
    self.aux_layer = nn.Sequential(nn.Linear(128 * ds_size ** 2, opt.n_classes), nn.Softmax())

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
    inputs = self.conv_blocks(inputs)
    inputs = inputs.view(inputs.shape[0], -1)
    if inputs.is_cuda and self.ngpu > 1:
      validity = nn.parallel.data_parallel(self.adv_layer, inputs, range(self.ngpu))
      label = nn.parallel.data_parallel(self.aux_layer, inputs, range(self.ngpu))
    else:
      validity = self.adv_layer(inputs)
      label = self.aux_layer(inputs)
    return validity, label


fixed_noise = torch.randn(opt.batch_size, nz, 1, 1, device=device)


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
  dataset = dset.FashionMNIST(root=opt.dataroot, download=True, transform=transforms.Compose([
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
  netG = Generator(ngpu).to(device)
  if opt.netG != '':
    netG.load_state_dict(torch.load(opt.netG, map_location=lambda storage, loc: storage))
  print(netG)

  netD = Discriminator(ngpu).to(device)
  if opt.netD != '':
    netD.load_state_dict(torch.load(opt.netD, map_location=lambda storage, loc: storage))
  print(netD)

  ################################################
  #           Loss functions
  ################################################
  adversarial_loss = torch.nn.BCELoss()
  auxiliary_loss = torch.nn.CrossEntropyLoss()

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
  print("Starting trainning!")
  for epoch in range(opt.n_epochs):
    for i, (data, labels) in enumerate(dataloader):
      # get data
      real_data = data.to(device)
      real_labels = labels.to(device)
      batch_size = real_data.size(0)

      # real data label is 1, fake data label is 0.
      valid = torch.full((batch_size,), 1, device=device)
      fake = torch.full((batch_size,), 0, device=device)
      # Sample noise as generator input
      noise = torch.randn(batch_size, nz, 1, 1, device=device)
      fake_labels = torch.tensor(np.random.randint(0, opt.n_classes, batch_size))

      ##############################################
      # (1) Update G network: maximize log(D(G(z)))
      ##############################################

      optimizerG.zero_grad()

      # Generate a batch of images
      fake_data = netG(noise, fake_labels)

      # Loss measures generator's ability to fool the discriminator
      validity, pred_label = netD(fake_data)
      loss_G = 0.5 * (adversarial_loss(validity, valid) + auxiliary_loss(pred_label, fake_labels))

      loss_G.backward()
      optimizerG.step()

      ##############################################
      # (2) Update D network: maximize log(D(x)) + log(1 - D(G(z)))
      ##############################################

      optimizerD.zero_grad()

      # Loss for real images
      real_pred, real_aux = netD(real_data)
      d_real_loss = (adversarial_loss(real_pred, valid) + auxiliary_loss(real_aux, real_labels)) / 2

      # Loss for fake images
      fake_pred, fake_aux = netD(fake_data.detach())
      d_fake_loss = (adversarial_loss(fake_pred, fake) + auxiliary_loss(fake_aux, fake_labels)) / 2

      # Total discriminator loss
      loss_D = (d_real_loss + d_fake_loss) / 2

      # Calculate discriminator accuracy
      pred = np.concatenate([real_aux.data.cpu().numpy(), fake_aux.data.cpu().numpy()], axis=0)
      gt = np.concatenate([labels.data.cpu().numpy(), fake_labels.data.cpu().numpy()], axis=0)
      acc_D = np.mean(np.argmax(pred, axis=1) == gt)

      loss_D.backward()
      optimizerD.step()

      print(f"Epoch->[{epoch + 1:03d}/{opt.n_epochs:03d}] "
            f"Progress->{i / len(dataloader) * 100:4.2f}% "
            f"Loss_D: {loss_D.item():.4f} "
            f"Acc_D: {100 * acc_D:.4f}% "
            f"Loss_G: {loss_G.item():.4f}", end="\r")

      if i % 100 == 0:
        vutils.save_image(real_data, f"{opt.out_images}/real_samples.png", normalize=True)
        with torch.no_grad():
          fake = netG(fixed_noise).detach().cpu()
        vutils.save_image(fake, f"{opt.out_images}/fake_samples_epoch_{epoch + 1:03d}.png", normalize=True)

    # do checkpointing
    torch.save(netG.state_dict(), f"{opt.checkpoints_dir}/netG_epoch_{epoch + 1:03d}.pth")
    torch.save(netD.state_dict(), f"{opt.checkpoints_dir}/netD_epoch_{epoch + 1:03d}.pth")


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
    create_gif("acgan.gif")
  elif opt.phase == 'generate':
    generate()
  else:
    print(opt)
