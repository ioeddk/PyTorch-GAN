import argparse
import os
import numpy as np
import math
from tqdm import tqdm
import torchvision.transforms as transforms
from torchvision.utils import save_image, make_grid

from torch.utils.data import DataLoader
from torchvision import datasets
from torch.autograd import Variable

import torch.nn as nn
import torch.nn.functional as F
import torch
from torch.utils.tensorboard import SummaryWriter

import deeplake

os.makedirs("images", exist_ok=True)
os.makedirs("logs", exist_ok=True)

# Initialize TensorBoard writer with timestamp
from datetime import datetime
current_time = datetime.now().strftime('%Y%m%d_%H%M%S')
writer = SummaryWriter(f'logs/tensorboard_{current_time}')

parser = argparse.ArgumentParser()
parser.add_argument("--n_epochs", type=int, default=200, help="number of epochs of training")
parser.add_argument("--batch_size", type=int, default=64, help="size of the batches")
parser.add_argument("--lr", type=float, default=0.0002, help="adam: learning rate")
parser.add_argument("--b1", type=float, default=0.5, help="adam: decay of first order momentum of gradient")
parser.add_argument("--b2", type=float, default=0.999, help="adam: decay of first order momentum of gradient")
parser.add_argument("--n_cpu", type=int, default=8, help="number of cpu threads to use during batch generation")
parser.add_argument("--latent_dim", type=int, default=100, help="dimensionality of the latent space")
parser.add_argument("--img_size", type=int, default=256, help="size of each image dimension")
parser.add_argument("--channels", type=int, default=3, help="number of image channels")
parser.add_argument("--sample_interval", type=int, default=400, help="interval between image sampling")
opt = parser.parse_args()
print(opt)

cuda = True if torch.cuda.is_available() else False

def weights_init_normal(m):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find("BatchNorm2d") != -1:
        torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
        torch.nn.init.constant_(m.bias.data, 0.0)


class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()

        self.init_size = opt.img_size // 16
        self.l1 = nn.Sequential(nn.Linear(opt.latent_dim, 128 * self.init_size ** 2))

        self.conv_blocks = nn.Sequential(
            nn.BatchNorm2d(128),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(128, 128, 3, stride=1, padding=1),
            nn.BatchNorm2d(128, 0.8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(128, 128, 3, stride=1, padding=1),
            nn.BatchNorm2d(128, 0.8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(128, 128, 3, stride=1, padding=1),
            nn.BatchNorm2d(128, 0.8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(128, 64, 3, stride=1, padding=1),
            nn.BatchNorm2d(64, 0.8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, opt.channels, 3, stride=1, padding=1),
            nn.Tanh(),
        )

    def forward(self, z):
        out = self.l1(z)
        out = out.view(out.shape[0], 128, self.init_size, self.init_size)
        img = self.conv_blocks(out)
        return img


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()

        def discriminator_block(in_filters, out_filters, bn=True):
            block = [nn.Conv2d(in_filters, out_filters, 3, 2, 1), nn.LeakyReLU(0.2, inplace=True), nn.Dropout2d(0.25)]
            if bn:
                block.append(nn.BatchNorm2d(out_filters, 0.8))
            return block

        self.model = nn.Sequential(
            *discriminator_block(opt.channels, 16, bn=False),
            *discriminator_block(16, 32),
            *discriminator_block(32, 64),
            *discriminator_block(64, 128),
            *discriminator_block(128, 256),
            *discriminator_block(256, 512),
        )

        # The height and width of downsampled image
        ds_size = opt.img_size // 2 ** 6
        self.adv_layer = nn.Sequential(nn.Linear(512 * ds_size ** 2, 1), nn.Sigmoid())

    def forward(self, img):
        out = self.model(img)
        out = out.reshape(out.shape[0], -1)
        validity = self.adv_layer(out)

        return validity


# Loss function
adversarial_loss = torch.nn.BCELoss()

# Initialize generator and discriminator
generator = Generator()
discriminator = Discriminator()

if cuda:
    generator.cuda()
    discriminator.cuda()
    adversarial_loss.cuda()

# Initialize weights
generator.apply(weights_init_normal)
discriminator.apply(weights_init_normal)

# Add variables to track best model
best_loss_balance = float('inf')
best_epoch = 0

# Configure data loader
# os.makedirs("../../data/mnist", exist_ok=True)
# dataloader = torch.utils.data.DataLoader(
#     datasets.MNIST(
#         "../../data/mnist",
#         train=True,
#         download=True,
#         transform=transforms.Compose(
#             [transforms.Resize(opt.img_size), transforms.ToTensor(), transforms.Normalize([0.5], [0.5])]
#         ),
#     ),
#     batch_size=opt.batch_size,
#     shuffle=True,
# )

ds = deeplake.load('hub://activeloop/wiki-art')

def transform(sample):
    # Get the image from the sample
    img = sample['images']
    # Convert to tensor and normalize
    img = torch.from_numpy(img).float()
    #permute to [C, H, W] format
    img = img.permute(2, 0, 1)
    # Resize to 256x256
    img = F.interpolate(img.unsqueeze(0), size=(opt.img_size, opt.img_size), mode='bilinear', align_corners=False).squeeze(0)
    # Normalize to [-1, 1]
    img = img / 255.0 * 2.0 - 1.0
    # Permute to [C, H, W] format
    # img = img.permute(2, 0, 1)
    return {'images': img, 'labels': sample['labels'], 'index': sample['index']}

def collate_fn(batch):
    # Stack images
    images = torch.stack([item['images'] for item in batch])
    # Stack labels
    labels = torch.stack([torch.tensor(item['labels']) for item in batch])
    # Stack indices
    indices = torch.stack([torch.tensor(item['index']) for item in batch])
    return {'images': images, 'labels': labels, 'index': indices}

dataloader = ds.pytorch(
    num_workers=0, 
    batch_size=opt.batch_size, 
    shuffle=False,
    transform=transform,
    collate_fn=collate_fn
)

# Optimizers
optimizer_G = torch.optim.Adam(generator.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))
optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))

Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor

# ----------
#  Training
# ----------

# for batch in dataloader:
#     print(batch)
#     exit()

for epoch in range(opt.n_epochs):
    all_gloss = []
    all_dloss = []
    for batch_idx, batch in tqdm(enumerate(dataloader)):
        # Extract images from the batch dictionary
        imgs = batch['images']
        
        # Images are already transformed, no need for additional processing
        real_imgs = Variable(imgs.type(Tensor))

        # Adversarial ground truths
        valid = Variable(Tensor(imgs.shape[0], 1).fill_(1.0), requires_grad=False)
        fake = Variable(Tensor(imgs.shape[0], 1).fill_(0.0), requires_grad=False)

        # -----------------
        #  Train Generator
        # -----------------

        optimizer_G.zero_grad()

        # Sample noise as generator input
        z = Variable(Tensor(np.random.normal(0, 1, (imgs.shape[0], opt.latent_dim))))

        # Generate a batch of images
        gen_imgs = generator(z)

        # Loss measures generator's ability to fool the discriminator
        g_loss = adversarial_loss(discriminator(gen_imgs), valid)

        g_loss.backward()
        optimizer_G.step()

        # ---------------------
        #  Train Discriminator
        # ---------------------

        optimizer_D.zero_grad()

        # Measure discriminator's ability to classify real from generated samples
        real_loss = adversarial_loss(discriminator(real_imgs), valid)
        fake_loss = adversarial_loss(discriminator(gen_imgs.detach()), fake)
        d_loss = (real_loss + fake_loss) / 2

        d_loss.backward()
        optimizer_D.step()

        all_gloss.append(g_loss.item())
        all_dloss.append(d_loss.item())

        # Log batch-level metrics to TensorBoard
        batches_done = epoch * len(dataloader) + batch_idx
        writer.add_scalar('Loss/Batch/Generator', g_loss.item(), batches_done)
        writer.add_scalar('Loss/Batch/Discriminator', d_loss.item(), batches_done)

        print(
            "[Epoch %d/%d] [Batch %d/%d] [D loss: %f] [G loss: %f]"
            % (epoch, opt.n_epochs, batch_idx, len(dataloader), d_loss.item(), g_loss.item())
        )

        batches_done = epoch * len(dataloader) + batch_idx
        if batches_done % opt.sample_interval == 0:
            save_image(gen_imgs.data[:25], "images/%d.png" % batches_done, nrow=5, normalize=True)
            # Log generated images to TensorBoard
            img_grid = make_grid(gen_imgs.data[:25], nrow=5, normalize=True)
            writer.add_image('Generated Images', img_grid, batches_done)
    
    # Save losses for this epoch
    gloss_epoch = sum(all_gloss) / len(all_gloss)
    dloss_epoch = sum(all_dloss) / len(all_dloss)
    
    # Log epoch-level metrics to TensorBoard
    writer.add_scalar('Loss/Epoch/Generator', gloss_epoch, epoch)
    writer.add_scalar('Loss/Epoch/Discriminator', dloss_epoch, epoch)
    
    # Calculate loss balance (absolute difference between G and D losses)
    loss_balance = abs(gloss_epoch - dloss_epoch)
    writer.add_scalar('Loss/Epoch/Balance', loss_balance, epoch)
    
    # Log model parameters histograms
    for name, param in generator.named_parameters():
        writer.add_histogram(f'Generator/{name}', param.data.cpu().numpy(), epoch)
    for name, param in discriminator.named_parameters():
        writer.add_histogram(f'Discriminator/{name}', param.data.cpu().numpy(), epoch)
    
    # Save best model if loss balance improves
    if loss_balance < best_loss_balance:
        best_loss_balance = loss_balance
        best_epoch = epoch
        # Save model checkpoints
        torch.save(generator.state_dict(), "logs/best_generator.pth")
        torch.save(discriminator.state_dict(), "logs/best_discriminator.pth")
        print(f"New best model saved at epoch {epoch} with loss balance: {loss_balance:.4f}")
    
    # Save losses to files
    with open("logs/gloss.txt", "a") as f:
        f.write(f"{epoch},{gloss_epoch}\n")
    with open("logs/dloss.txt", "a") as f:
        f.write(f"{epoch},{dloss_epoch}\n")

# Save final model checkpoint
torch.save(generator.state_dict(), "logs/final_generator.pth")
torch.save(discriminator.state_dict(), "logs/final_discriminator.pth")
print(f"Final model saved at epoch {epoch}")

# Close TensorBoard writer
writer.close()
