from utils import *

#Optional
class Autoencoder(nn.Module):
    # Class with encoder and decoder model for the GAN part of the model
    def __init__(self):
        super(Autoencoder, self).__init__()
        
        # Encoder
        # Used to take input data from GAN and extract key features
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1), # assuming input is grayscale (1 channel)
            nn.ReLU(True),
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(True),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0), # Subsampling
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(True),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0), # Subsampling
            nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1),
            nn.ReLU(True)
        )
        
        # Decoder
        # 
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2, padding=0), # Deconv (upsampling)
            nn.ReLU(True),
            nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2, padding=0), # Deconv (upsampling)
            nn.ReLU(True),
            nn.Conv2d(128, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(True),
            nn.Conv2d(64, 1, kernel_size=3, stride=1, padding=1),
            nn.Sigmoid()  # to ensure output is in range [0, 1]
        )
        
    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x