import torch
import torch.nn as nn
import torch.nn.functional as F

# Function to calculate total parameters
def count_parameters(model):
   return sum(p.numel() for p in model.parameters() if p.requires_grad)

class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2 for arbitrary dimensions"""

    def __init__(self, dim, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels

        ConvLayer = getattr(nn, f'Conv{dim}d')
        BatchNormLayer = getattr(nn, f'BatchNorm{dim}d')
        
        self.double_conv = nn.Sequential(
            ConvLayer(in_channels, mid_channels, kernel_size=3, padding=1, bias=False, padding_mode='circular'),
            BatchNormLayer(mid_channels),
            nn.ReLU(inplace=True),
            ConvLayer(mid_channels, out_channels, kernel_size=3, padding=1, bias=False,  padding_mode='circular'),
            BatchNormLayer(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)


class Down(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, dim, in_channels, out_channels):
        super().__init__()
        MaxPoolLayer = getattr(nn, f'MaxPool{dim}d')
        
        self.maxpool_conv = nn.Sequential(
            MaxPoolLayer(2),
            DoubleConv(dim, in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class Up(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, dim, in_channels, out_channels, bilinear=True):
        super().__init__()

        if bilinear:
            UpsampleLayer = nn.Upsample
            scale_factor = 2
            self.up = UpsampleLayer(scale_factor=scale_factor, mode=f'bilinear' if dim == 2 else 'trilinear', align_corners=True)
            self.conv = DoubleConv(dim, in_channels, out_channels, in_channels // 2)
        else:
            ConvTransposeLayer = getattr(nn, f'ConvTranspose{dim}d')
            self.up = ConvTransposeLayer(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(dim, in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        # Determine padding to ensure the dimensions match after upsampling
        diff_dims = [x2.size(i) - x1.size(i) for i in range(len(x1.size())-1,1,-1)]
        # pad = []
        # for d in diff_dims:
        #     pad = pad + [d // 2] + [d - (d // 2)]
        pad = [item for d in diff_dims for item in (d // 2, d - d // 2)]
        # print(diff_dims)
        # print(pad)
        # print(pad)
        x1 = F.pad(x1, pad)
        # print(x1.shape)
        # print(x2.shape)
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)

class Up_new(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, dim, in_channels, out_channels, bilinear=True):
        super().__init__()

        if bilinear:
            UpsampleLayer = nn.Upsample
            scale_factor = 2
            self.up = UpsampleLayer(scale_factor=scale_factor, mode=f'bilinear' if dim == 2 else 'trilinear', align_corners=True)
            self.conv = DoubleConv(dim, in_channels, out_channels, in_channels // 2)
        else:
            ConvTransposeLayer = getattr(nn, f'ConvTranspose{dim}d')
            self.up = ConvTransposeLayer(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(dim, in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        # # Determine padding to ensure the dimensions match after upsampling
        # diff_dims = [x2.size(i) - x1.size(i) for i in range(len(x1.size())-1,1,-1)]
        # # pad = []
        # # for d in diff_dims:
        # #     pad = pad + [d // 2] + [d - (d // 2)]
        # pad = [item for d in diff_dims for item in (d // 2, d - d // 2)]
        # # print(diff_dims)
        # # print(pad)
        # print(pad)
        # x1 = F.pad(x1, pad)
        # print(x1.shape)
        # print(x2.shape)
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)
    
class OutConv(nn.Module):
    """Final Convolutional Layer"""

    def __init__(self, dim, in_channels, out_channels):
        super(OutConv, self).__init__()
        ConvLayer = getattr(nn, f'Conv{dim}d')
        self.conv = ConvLayer(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)


class UNet_base(nn.Module):
    def __init__(self, dim, n_channels, n_classes, bilinear=False):
        super(UNet_base, self).__init__()
        self.dim = dim
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear
        if dim == 2:
            self.inc = DoubleConv(dim, n_channels, 64)
            self.down1 = Down(dim, 64, 128)
            self.down2 = Down(dim, 128, 256)
            self.down3 = Down(dim, 256, 512)
            factor = 2 if bilinear else 1
            self.down4 = Down(dim, 512, 1024 // factor)
            self.up1 = Up(dim, 1024, 512 // factor, bilinear)
            self.up2 = Up(dim, 512, 256 // factor, bilinear)
            self.up3 = Up(dim, 256, 128 // factor, bilinear)
            self.up4 = Up(dim, 128, 64, bilinear)
            self.outc = OutConv(dim, 64, n_classes)
        else:
            self.inc = DoubleConv(dim, n_channels, 16)
            self.down1 = Down(dim, 16, 32)
            self.down2 = Down(dim, 32, 64)
            self.down3 = Down(dim, 64, 128)
            factor = 2 if bilinear else 1
            self.down4 = Down(dim, 128, 256 // factor)
            self.up1 = Up(dim, 256, 128 // factor, bilinear)
            self.up2 = Up(dim, 128, 64 // factor, bilinear)
            self.up3 = Up(dim, 64, 32 // factor, bilinear)
            self.up4 = Up(dim, 32, 16, bilinear)
            self.outc = OutConv(dim, 16, n_classes)            

    def forward(self, x):
        if self.dim == 1 and len(x.shape) == 2:
            x = x.unsqueeze(1)
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.outc(x)
        return logits
    
class UNet(nn.Module):
    def __init__(self, dim, n_channels, n_classes, bilinear=False):
        super(UNet, self).__init__()
        self.dim = dim
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        self.inc = DoubleConv(dim, n_channels, 8)
        self.down1 = Down(dim, 8, 16)
        self.down2 = Down(dim, 16, 32)
        self.down3 = Down(dim, 32, 64)
        factor = 2 if bilinear else 1
        self.down4 = Down(dim, 64, 128 // factor)
        self.up1 = Up_new(dim, 128, 64 // factor, bilinear)
        self.up2 = Up_new(dim, 64, 32 // factor, bilinear)
        self.up3 = Up_new(dim, 32, 16 // factor, bilinear)
        self.up4 = Up_new(dim, 16, 8, bilinear)
        self.outc = OutConv(dim, 8, n_classes)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.outc(x)
        return logits
    
def main():
    # Create a model instance
    # model = UNet_base(dim=3, n_channels=1, n_classes=1)
    model = UNet(dim=1, n_channels=1, n_classes=1)
    # Calculate total parameters
    total_params = count_parameters(model)
    print("Total trainable parameters:", total_params) 
    # Generate a random tensor to simulate input data
    spec  = torch.randn(3,1,256)
    # Forward pass through the network
    output = model(spec)

    # Print the output
    print("Output of the network:", output.size())

if __name__ == "__main__":
    main()

