import torch
import torch.nn as nn
import torch.nn.functional as F

# Ignore this Class, use one that's outlined in a later block

class SelfAttention(nn.Module):
    def __init__(self, in_channels):
        super(SelfAttention, self).__init__()
        # Define the number of channels for the intermediate convolutions
        self.f_channels = in_channels // 32
        self.g_channels = in_channels // 32
        self.h_channels = in_channels //16  # Reduced from in_channels to in_channels // 2
        
        # Define the 1x1 convolutions
        self.f_conv = nn.Conv2d(in_channels, self.f_channels, kernel_size=1, stride=1, padding=0)
        self.g_conv = nn.Conv2d(in_channels, self.g_channels, kernel_size=1, stride=1, padding=0)
        self.h_conv = nn.Conv2d(in_channels, self.h_channels, kernel_size=1, stride=1, padding=0)
        
        # Define the final 1x1 convolution
        self.v_conv = nn.Conv2d(self.h_channels, in_channels, kernel_size=1, stride=1, padding=0)
        
        # Initialize the trainable parameter lambda
        self.lambda_param = nn.Parameter(torch.zeros(1))
        
    def forward(self, x):
        batch_size, C, W, H = x.size()
        
        # Apply the convolutions
        f = self.f_conv(x)
        g = self.g_conv(x)
        h = self.h_conv(x)
        
        # Reshape for matrix multiplication
        f = f.view(batch_size, self.f_channels, -1)
        g = g.view(batch_size, self.g_channels, -1)
        h = h.view(batch_size, self.h_channels, -1)
        
        # Transpose f for the dot product
        f = f.permute(0, 2, 1)
        
        # Calculate the attention map
        attention_map = torch.bmm(f, g)
        attention_map = F.softmax(attention_map, dim=-1)
        
        # Apply the attention map to h
        o = torch.bmm(h, attention_map.permute(0, 2, 1))
        
        # Reshape the output to the original dimensions
        o = o.view(batch_size, self.h_channels, W, H)
        
        # Apply the final 1x1 convolution to o
        v = self.v_conv(o)
        
        # Multiply by the trainable parameter and add the input
        y = self.lambda_param * v + x
        
        return y



# import torch
# import torch.nn as nn
# import torch.nn.functional as F

# # TODO: Run this code
# # Remove the previous one? 34:00
# # Self Attention: Useful for capturing long-range information in the data
# # Used in the UNet Convolutional Layers


# class SelfAttention(nn.Module):
#     def __init__(self, in_channels):
#         super(SelfAttention, self).__init__()
#         # Define the number of channels for the intermediate convolutions
#         self.f_channels = in_channels // 32 # 32
#         self.g_channels = in_channels // 32 # 32
#         self.h_channels = in_channels // 16 # 16 # Reduced from in_channels to in_channels // 2
        
#         # Define the 1x1 convolutions
#         self.f_conv = nn.Conv2d(in_channels, self.f_channels, kernel_size=1, stride=1, padding=0)
#         self.g_conv = nn.Conv2d(in_channels, self.g_channels, kernel_size=1, stride=1, padding=0)
#         self.h_conv = nn.Conv2d(in_channels, self.h_channels, kernel_size=1, stride=1, padding=0)
        
#         # Define the final 1x1 convolution
#         self.v_conv = nn.Conv2d(self.h_channels, in_channels, kernel_size=1, stride=1, padding=0)
        
#         # Initialize the trainable parameter lambda
#         self.lambda_param = nn.Parameter(torch.zeros(1))
        
#     def forward(self, x):
#         batch_size, C, W, H = x.size()
        
#         # Apply the convolutions
#         f = self.f_conv(x)
#         g = self.g_conv(x)
#         h = self.h_conv(x)
        
#         # Reshape for matrix multiplication
#         f = f.view(batch_size, self.f_channels, -1)
#         g = g.view(batch_size, self.g_channels, -1)
#         h = h.view(batch_size, self.h_channels, -1)
        
#         # Transpose f for the dot product
#         f = f.permute(0, 2, 1)
        
#         # Calculate the attention map
#         a "__main__":
#     input_tensor = torch.randn(1, 64, 32, 32).cuda()  # Batch size of 1, 64 channels, 32x32 feature map
#     self_attention = SelfAttention(64).cuda()
#     output_tensor = self_attention(input_tensor)
#     print(output_tensor.shape)
#         attention_map = torch.bmm(f, g)
#         attention_map = F.softmax(attention_map, dim=-1)
        
#         # Apply the attention map to h
#         o = torch.bmm(h, attention_map.permute(0, 2, 1))
        
#         # Reshape the output to the original dimensions
#         o = o.view(batch_size, self.h_channels, W, H)
        
#         # Apply the final 1x1 convolution to o
#         v = self.v_conv(o)
        
#         # Multiply by the trainable parameter and add the input
#         y = self.lambda_param * v + x
        
#         return y


# # Example usage
# #if __name__ ==