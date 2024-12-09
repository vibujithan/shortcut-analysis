import torch.nn as nn
from monai.networks.blocks import Convolution, UpSample
from monai.networks.layers.factories import Act, Norm


class Encoder(nn.Module):
    def __init__(self, in_channels=1, latent_size=512):
        super().__init__()
        self.latent_size = latent_size

        self.encoder_layers = nn.Sequential(
            # Layer 1: 180x180x180 -> 90x90x90
            Convolution(
                spatial_dims=3,
                in_channels=in_channels,
                out_channels=32,
                kernel_size=3,
                strides=2,
                padding=0,
                act=Act.RELU,
                norm=Norm.BATCH
            ),

            # Layer 2: 90x90x90 -> 45x45x45
            Convolution(
                spatial_dims=3,
                in_channels=32,
                out_channels=64,
                kernel_size=3,
                strides=2,
                padding=1,
                act=Act.RELU,
                norm=Norm.BATCH
            ),

            # Layer 3: 45x45x45 -> 23x23x23
            Convolution(
                spatial_dims=3,
                in_channels=64,
                out_channels=128,
                kernel_size=3,
                strides=2,
                padding=0,
                act=Act.RELU,
                norm=Norm.BATCH
            ),

            # Layer 4: 23x23x23 -> 12x12x12
            Convolution(
                spatial_dims=3,
                in_channels=128,
                out_channels=256,
                kernel_size=3,
                strides=2,
                padding=1,
                act=Act.RELU,
                norm=Norm.BATCH
            ),
        )

        # Flatten and project to latent space
        self.flatten = nn.Flatten()
        self.fc = nn.Linear(256 * 11 * 11 * 11, latent_size)

    def forward(self, x):
        x = self.encoder_layers(x)
        x = self.flatten(x)
        return self.fc(x)

    def get_output_size(self):
        return self.latent_size


class Decoder(nn.Module):
    def __init__(self, latent_size=512, out_channels=1):
        super().__init__()

        self.fc = nn.Linear(latent_size, 256 * 11 * 11 * 11)
        self.unflatten = nn.Unflatten(1, (256, 11, 11, 11))

        self.decoder_layers = nn.Sequential(
            # Layer 1: 12x12x12 -> 23x23x23
            UpSample(
                spatial_dims=3,
                in_channels=256,
                out_channels=128,
                scale_factor=2,
                mode='deconv'
            ),
            Convolution(
                spatial_dims=3,
                in_channels=128,
                out_channels=128,
                kernel_size=3,
                padding=1,
                act=Act.RELU,
                norm=Norm.BATCH
            ),

            # Layer 2: 23x23x23 -> 45x45x45
            UpSample(
                spatial_dims=3,
                in_channels=128,
                out_channels=64,
                scale_factor=2,
                mode='deconv'
            ),
            Convolution(
                spatial_dims=3,
                in_channels=64,
                out_channels=64,
                kernel_size=3,
                padding=1,
                act=Act.RELU,
                norm=Norm.BATCH
            ),

            # Layer 3: 45x45x45 -> 90x90x90
            UpSample(
                spatial_dims=3,
                in_channels=64,
                out_channels=32,
                scale_factor=2,
                mode='deconv'
            ),
            Convolution(
                spatial_dims=3,
                in_channels=32,
                out_channels=32,
                kernel_size=3,
                padding=2,
                act=Act.RELU,
                norm=Norm.BATCH
            ),

            # Layer 4: 90x90x90 -> 180x180x180
            UpSample(
                spatial_dims=3,
                in_channels=32,
                out_channels=32,
                scale_factor=2,
                mode='deconv'
            ),
            Convolution(
                spatial_dims=3,
                in_channels=32,
                out_channels=out_channels,
                kernel_size=3,
                padding=1,
                act=None,
                norm=None
            )
        )

    def forward(self, z):
        x = self.fc(z)
        x = self.unflatten(x)
        x = self.decoder_layers(x)
        return x


class Autoencoder3D(nn.Module):
    def __init__(self, in_channels=1, latent_size=512):
        super().__init__()
        self.encoder = Encoder(in_channels=in_channels, latent_size=latent_size)
        self.decoder = Decoder(latent_size=latent_size, out_channels=in_channels)

    def forward(self, x):
        z = self.encoder(x)
        return self.decoder(z)
