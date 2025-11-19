from utils import *

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Sobel Edge Detection to apply filter to get edges within an image
class SobelEdgeDetection(nn.Module):
    def __init__(self):
        super(SobelEdgeDetection, self).__init__()
        sobel_kernel_x = torch.tensor([[-1, 0, 1],
                                      [-2, 0, 2],
                                      [-1, 0, 1]], dtype=torch.float32).view(1, 1, 3, 3)
        sobel_kernel_y = torch.tensor([[-1, -2, -1],
                                      [0, 0, 0],
                                      [1, 2, 1]], dtype=torch.float32).view(1, 1, 3, 3)
        
        sobel_kernel_x = sobel_kernel_x.to(device)
        sobel_kernel_y = sobel_kernel_y.to(device)
        
        # Extend Sobel kernels for 1 channel (RGB images)
        self.sobel_kernel_x = torch.cat([sobel_kernel_x] * 1, dim=1)
        self.sobel_kernel_y = torch.cat([sobel_kernel_y] * 1, dim=1)

    def forward(self, x):
        # Apply Sobel convolution along x and y axes
        sobel_x_output = F.conv2d(x, self.sobel_kernel_x, padding=1)
        sobel_y_output = F.conv2d(x, self.sobel_kernel_y, padding=1)
        
        # Calculate edge magnitude
        edge_magnitude = torch.sqrt(sobel_x_output.pow(2) + sobel_y_output.pow(2))
        edge_magnitude = edge_magnitude/edge_magnitude.max() 
        return edge_magnitude

class SobelConvolutionNet(nn.Module):
    def __init__(self):
        super(SobelConvolutionNet, self).__init__()
       
        self.sobel = SobelEdgeDetection()
        self.conv1 = nn.Conv2d(1, 64, kernel_size=3, stride = 1, padding=1)
        self.conv2 = nn.Conv2d(64, 32, kernel_size=3,stride = 1, padding=1)
        self.conv2D = nn.ConvTranspose2d( in_channels = 32, out_channels = 32, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(32, 1, kernel_size=3, stride =1, padding=1)
        
    def forward (self,x):
        x = torch.relu(self.sobel(x))
        x = torch.relu((self.conv1 (x)))
        x = torch.relu((self.conv2 (x)))
        x = torch.relu(self.conv2D (x))
        
        out = self.conv3 (x)
        out = torch.relu(out)
        
        return out