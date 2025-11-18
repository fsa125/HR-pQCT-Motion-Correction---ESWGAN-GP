from utils import *
from SobelEdgeDetection import *
from SelfAttention import SelfAttention

# acts as a human observer -- takes first few layers and predicts score of image

class FeatureExtractor(nn.Module):
    def __init__(self):
        super(FeatureExtractor, self).__init__()
        vgg19_model = vgg19(pretrained=True)
        # Modify vgg19 features for 1 channel input
        vgg_features = list(vgg19_model.features.children())
        vgg_features[0] = nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1)
        self.feature_extractor = nn.Sequential(*vgg_features[:18])

    def forward(self, img):
        return self.feature_extractor(img)
    
    
# Extract Features from the convolutional layers (other than the first and last layers)    
class InceptionFeatureExtractor(nn.Module):
    def __init__(self):
        super(InceptionFeatureExtractor, self).__init__()
        
        # Load pre-trained Inception v3 model
        inception_model = inception_v3(pretrained=True, transform_input=False)
        
        # Extract features up to the last convolutional layer (before the fully connected layers)
        # Inception v3 has an auxiliary output which we don't need
        
        # If using a single channel input, you might need to adjust the first convolution layer
        # Inception v3 starts with a 3-channel input, so this step is optional
        
        first_conv = nn.Conv2d(1, 32, kernel_size=3, stride=2, padding=0)
        
        # Create a new sequential model with the modified initial convolution
        self.feature_extractor = nn.Sequential(
            first_conv,
            *list(inception_model.children())[1:-2]  # Exclude the initial convolution and last two layers
        )
    
    def forward(self, img):
        return self.feature_extractor(img)


class UNetGenerator(nn.Module):
    # This class is not used, look at the UNetAttGenerator Class
    def __init__(self, in_channels=1, out_channels=1):
        super(UNetGenerator, self).__init__()

        def conv_block(in_channels, out_channels, kernel_size=3, stride=1, padding=1, batch_norm=False):
            layers = [nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)]
            if batch_norm:
                layers.append(nn.BatchNorm2d(out_channels))
            layers.append(nn.ReLU(inplace=True))
            return nn.Sequential(*layers)

        # Encoder (downsampling)
        self.enc1 = conv_block(in_channels, 64)
        self.enc2 = conv_block(64, 128, stride=2)
        self.enc3 = conv_block(128, 256, stride=2)
        self.enc4 = conv_block(256, 512, stride=2)

        # Decoder (upsampling)
        self.dec1 = nn.Sequential(conv_block(512, 256), nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False))
        self.dec2 = nn.Sequential(conv_block(512, 128), nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False))
        self.dec3 = nn.Sequential(conv_block(256, 64), nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False))
        self.dec4 = conv_block(128, out_channels)
        self.edge = SobelConvolutionNet()

    def forward(self, x):
        # Encoder
        enc1 = self.enc1(x)
        enc2 = self.enc2(enc1)
        enc3 = self.enc3(enc2)
        enc4 = self.enc4(enc3)

        # Decoder with skip connections
        dec1 = self.dec1(enc4)
        dec1 = torch.cat((dec1, enc3), dim=1)
        dec2 = self.dec2(dec1)
        dec2 = torch.cat((dec2, enc2), dim=1)
        dec3 = self.dec3(dec2)
        dec3 = torch.cat((dec3, enc1), dim=1)
        dec4 = self.dec4(dec3)
        ### Modifications
        #dec4 = torch.cat((dec4,x),dim = 1)
        #dec4 = self.dec4(dec4)
        dec4 = dec4 + x + self.edge(x)
        ### Modifications
        
        return dec4

# this is the UNet Class that is used for the model
class UNetAttGenerator(nn.Module):
    # represented by Brown, Teal, and Light blue blocks in the paper
    def __init__(self, in_channels=1, out_channels=1):
        super(UNetAttGenerator, self).__init__()

        def conv_block(in_channels, out_channels, kernel_size=3, stride=1, padding=1, batch_norm=False):
            layers = [nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)]
            if batch_norm:
                layers.append(nn.BatchNorm2d(out_channels))
            layers.append(nn.ReLU(inplace=True))
            return nn.Sequential(*layers)

        # Encoder (downsampling)
        self.enc1 = conv_block(in_channels, 64)
        self.enc2 = conv_block(64, 128, stride=2)
        self.enc3 = conv_block(128, 256, stride=2)
        self.att3 = SelfAttention(256)
        self.enc4 = conv_block(256, 512, stride=2)
        self.att4 = SelfAttention(512)

        # Decoder (upsampling)
        self.dec1 = nn.Sequential(conv_block(512, 256), nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False))
        self.attdec1 = SelfAttention(512)
        self.dec2 = nn.Sequential(conv_block(512, 128), nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False))
        self.attdec2 = SelfAttention(256)
        self.dec3 = nn.Sequential(conv_block(256, 64), nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False))
        self.dec4 = conv_block(128, out_channels)
        self.edge = SobelConvolutionNet()

    def forward(self, x):
        # Encoder
        enc1 = self.enc1(x)
        enc2 = self.enc2(enc1)
        enc3 = self.enc3(enc2)
        enc3 = self.att3(enc3)
        enc4 = self.enc4(enc3)
        enc4 = self.att4(enc4)

        # Decoder with skip connections
        dec1 = self.dec1(enc4)
        dec1 = torch.cat((dec1, enc3), dim=1)
        dec1 = self.attdec1 (dec1) #Added
        dec2 = self.dec2(dec1)
        dec2 = torch.cat((dec2, enc2), dim=1)
        dec2 = self.attdec2 (dec2)
        
        dec3 = self.dec3(dec2)
        dec3 = torch.cat((dec3, enc1), dim=1)
        dec4 = self.dec4(dec3)
        
        # this is the line that connects UNet to the GAN model
        # dec4 = output of the UNet Model image
        # x = original image that's added as an input to the UNet model
        # self.edge(x) = output from the sobel edge detection (applied to image that's an input to UNet model)
        dec4 = dec4 + x + self.edge(x)

        return dec4
    
    
class Discriminator(nn.Module):
    # This class is not used in the model, use the below one
    def __init__(self, input_shape):
        super(Discriminator, self).__init__()

        self.input_shape = input_shape
        in_channels, in_height, in_width = self.input_shape
        patch_h, patch_w = int(in_height / 2 ** 4), int(in_width / 2 ** 4)
        self.output_shape = (1, patch_h, patch_w)

        def discriminator_block(in_filters, out_filters, first_block=False):
            layers = []
            layers.append(nn.Conv2d(in_filters, out_filters, kernel_size=3, stride=1, padding=1))
            if not first_block:
                layers.append(nn.BatchNorm2d(out_filters))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            layers.append(nn.Conv2d(out_filters, out_filters, kernel_size=3, stride=2, padding=1))
            layers.append(nn.BatchNorm2d(out_filters))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers

        layers = []
        in_filters = in_channels
        for i, out_filters in enumerate([64, 128, 256, 512]):
            layers.extend(discriminator_block(in_filters, out_filters, first_block=(i == 0)))
            in_filters = out_filters
            

        layers.append(nn.Conv2d(out_filters, 1, kernel_size=3, stride=1, padding=1))

        self.model = nn.Sequential(*layers)

    def forward(self, img):
        return self.model(img)

# Use this class for the model
class DiscriminatorATT(nn.Module):
    # Discriminator for the GAN part of the model
    # Green blocks from the paper
    
    def __init__(self, input_shape):
        super(DiscriminatorATT, self).__init__()

        self.input_shape = input_shape
        in_channels, in_height, in_width = self.input_shape
        patch_h, patch_w = int(in_height / 2 ** 4), int(in_width / 2 ** 4)
        self.output_shape = (1, patch_h, patch_w)
        self.self_attention = SelfAttention(512)

        def discriminator_block(in_filters, out_filters, first_block=False):
            layers = []
            layers.append(nn.Conv2d(in_filters, out_filters, kernel_size=3, stride=1, padding=1))
            if not first_block:
                layers.append(nn.BatchNorm2d(out_filters))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            layers.append(nn.Conv2d(out_filters, out_filters, kernel_size=3, stride=2, padding=1))
            layers.append(nn.BatchNorm2d(out_filters))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers

        layers = []
        in_filters = in_channels
        for i, out_filters in enumerate([64, 128, 256, 512]):
            layers.extend(discriminator_block(in_filters, out_filters, first_block=(i == 0)))
            in_filters = out_filters
            
        layers.append(self.self_attention)
        layers.append(nn.Conv2d(out_filters, 1, kernel_size=3, stride=1, padding=1))

        self.model = nn.Sequential(*layers)

    def forward(self, img):
        return self.model(img)