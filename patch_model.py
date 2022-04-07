import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init
import math



class conv_block(nn.Module):
    def __init__(self, input_channel, output_channel, kernel_size, stride=1, padding=1):
        super(conv_block, self).__init__()
        self.conv1 = nn.Conv3d(input_channel, output_channel, kernel_size=kernel_size, stride=stride, padding=padding)
        self.batch_norm = nn.BatchNorm3d(output_channel)
        self.relu =  nn.ReLU(inplace=True)
    def forward(self, x):
        x = self.relu(self.batch_norm(self.conv1(x)))
        return x

class resnet_block(nn.Module):
    def __init__(self, channels, kernel_size, stride=1, padding=1):
        super(resnet_block, self).__init__()
        self.layers = nn.Sequential(
            nn.Conv3d(channels, channels, kernel_size=kernel_size, stride=stride, padding=padding),
            nn.BatchNorm3d(channels),
            nn.ReLU(inplace=True),
            nn.Conv3d(channels, channels, kernel_size=kernel_size, stride=stride, padding=padding),
            nn.BatchNorm3d(channels),
            nn.ReLU(inplace=True),
        )
    def forward(self, x):
        x = self.layers(x) + x
        return x


class upsample_conv(nn.Module):
    def __init__(self, input_channels, output_channels, size, kernel_size=1, align_corners=False):
        super(upsample_conv, self).__init__()
        self.up = nn.Sequential(
            nn.Conv3d(input_channels, output_channels, kernel_size=kernel_size),
            nn.Upsample(size,  mode='trilinear', align_corners=align_corners),
        )
    def forward(self, x):
        return self.up(x)

class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        self.conv1 = conv_block(input_channel=1, output_channel=32, kernel_size=3)
        self.res1 = resnet_block(channels=32, kernel_size=3)
        self.pool1 = nn.MaxPool3d(3, stride=2, padding=1)

        self.conv2 = conv_block(input_channel=32, output_channel=64, kernel_size=3)
        self.res2 = resnet_block(channels=64, kernel_size=3)
        self.pool2 = nn.MaxPool3d(3, stride=2, padding=1)

        self.conv3 = conv_block(input_channel=64, output_channel=128, kernel_size=3)
        self.res3 = resnet_block(channels=128, kernel_size=3)
        self.pool3 = nn.MaxPool3d(3, stride=2, padding=1)

        self.conv4 = conv_block(input_channel=128, output_channel=256, kernel_size=3)
        self.res4 = resnet_block(channels=256, kernel_size=3)
        self.pool4 = nn.MaxPool3d(3, stride=2, padding=1)

        # self.conv5 = conv_block(input_channel=256, output_channel=512, kernel_size=3)
        # self.res5 = resnet_block(channels=512, kernel_size=3)
        # self.pool5 = nn.MaxPool3d(3, stride=2, padding=1)

    def forward(self, x):
        x = self.pool1(self.res1(self.conv1(x)))
        x = self.pool2(self.res2(self.conv2(x)))
        x = self.pool3(self.res3(self.conv3(x)))
        x = self.pool4(self.res4(self.conv4(x)))
        # if(resolution<=2):
        #     x = self.pool5(self.res5(self.conv5(x)))
        return x

class Decoder(nn.Module):
    """ Decoder Module """
    def __init__(self, latent_dim):
        super(Decoder, self).__init__()
        self.latent_dim = latent_dim
        self.linear_up = nn.Linear(latent_dim, 256)
        self.relu = nn.ReLU()
        self.upsize5 = upsample_conv(input_channels=256, output_channels=128, size=(2, 2, 2), kernel_size=1)
        self.res_block5 = resnet_block(channels=128, kernel_size=3)
        self.upsize4 = upsample_conv(input_channels=128, output_channels=64, size=(4,4,4), kernel_size=1)
        self.res_block4 = resnet_block(channels=64, kernel_size=3)
        self.upsize3 = upsample_conv(input_channels=64, output_channels=32, size=(8,8,8), kernel_size=1)
        self.res_block3 = resnet_block(channels=32, kernel_size=3)
        self.upsize2 = upsample_conv(input_channels=32, output_channels=1, size=(16,16,16), kernel_size=1)
        self.res_block2 = resnet_block(channels=1, kernel_size=3)
        #self.upsize1 = upsample_conv(input_channels=32, output_channels=1, size=(118,136,118), kernel_size=1)
        #self.res_block1 = resnet_block(channels=1, kernel_size=3)


    def forward(self, x):
        x = self.relu(self.linear_up(x))
        x = x.view(-1, 256, 1, 1, 1)
        x = self.upsize5(x)
        x = self.res_block5(x)
        x = self.upsize4(x)
        x = self.res_block4(x)
        x = self.upsize3(x)
        x = self.res_block3(x)
        x = self.upsize2(x)
        x = self.res_block2(x)
        # if(resolution<=2):
        #     x = self.upsize1(x)
        #     x = self.res_block1(x)
        return x


class VAE(nn.Module):
    def __init__(self, latent_dim=128):
        super(VAE, self).__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.latent_dim = latent_dim
        self.z_mean = nn.Linear(256, latent_dim)
        self.z_log_sigma = nn.Linear(256, latent_dim)
        self.epsilon = torch.normal(size=(1, latent_dim), mean=0, std=1.0, device=self.device)
        self.encoder = Encoder()
        self.decoder = Decoder(latent_dim)

    def forward(self, x):
        x = self.encoder(x)
        x = torch.flatten(x, start_dim=1)
        z_mean = self.z_mean(x)
        z_log_sigma = self.z_log_sigma(x)
        z = z_mean + z_log_sigma.exp()*self.epsilon
        y = self.decoder(z)
        return y, z_mean, z_log_sigma
