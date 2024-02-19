### Ecosystem Imports ###
import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), "."))
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

### External Imports ###
import math
import torch as tc
import torch.nn.functional as F

### Internal Imports ###


########################


def pad(image : tc.Tensor, template : tc.Tensor) -> tc.Tensor:
    pad_x = math.fabs(image.size(3) - template.size(3))
    pad_y = math.fabs(image.size(2) - template.size(2))
    pad_z = math.fabs(image.size(4) - template.size(4))
    b_x, e_x = math.floor(pad_x / 2), math.ceil(pad_x / 2)
    b_y, e_y = math.floor(pad_y / 2), math.ceil(pad_y / 2)
    b_z, e_z = math.floor(pad_z / 2), math.ceil(pad_z / 2)
    image = F.pad(image, (b_z, e_z, b_x, e_x, b_y, e_y))
    return image

def pad_with_template(image : tc.Tensor, template : tc.Tensor) -> tuple[tc.Tensor, tc.Tensor]:
    pad_x = math.fabs(image.size(3) - template.size(3))
    pad_y = math.fabs(image.size(2) - template.size(2))
    pad_z = math.fabs(image.size(4) - template.size(4))
    b_x, e_x = math.floor(pad_x / 2), math.ceil(pad_x / 2)
    b_y, e_y = math.floor(pad_y / 2), math.ceil(pad_y / 2)
    b_z, e_z = math.floor(pad_z / 2), math.ceil(pad_z / 2)
    if image.size(3) < template.size(3):
        padded_image = F.pad(image, (0, 0, b_x, e_x))
        padded_template = template
    elif image.size(3) > template.size(3):
        padded_image = image
        padded_template = F.pad(template, (0, 0, b_x, e_x))
    else:
        padded_image = image
        padded_template = template
    if image.size(2) < template.size(2):
        padded_image = F.pad(padded_image, (0, 0, 0, 0, b_y, e_y))
    elif image.size(2) > template.size(2):
        padded_template = F.pad(padded_template, (0, 0, 0, 0, b_y, e_y))
    else:
        pass
    if image.size(4) < template.size(4):
        padded_image = F.pad(padded_image, (b_z, e_z))
    elif image.size(4) > template.size(4):
        padded_template = F.pad(padded_template, (b_z, e_z))
    else:
        pass    
    return padded_image, padded_template

def resample(x : tc.Tensor, template : tc.Tensor) -> tc.Tensor:
    return F.interpolate(x, template.shape[2:], mode='trilinear')

class ConvolutionalBlock(tc.nn.Module):
    def __init__(self, input_size : int, output_size : int, leaky_alpha : float=0.01):
        super(ConvolutionalBlock, self).__init__()

        self.module = tc.nn.Sequential(
            tc.nn.Conv3d(input_size, output_size, 3, stride=1, padding=1),
            tc.nn.GroupNorm(output_size, output_size),
            tc.nn.LeakyReLU(leaky_alpha, inplace=True),
            tc.nn.Conv3d(output_size, output_size, 3, stride=1, padding=1),
            tc.nn.GroupNorm(output_size, output_size),
            tc.nn.LeakyReLU(leaky_alpha, inplace=True),        
        )

    def forward(self, x : tc.Tensor):
        return self.module(x)
    

class ResidualBlock(tc.nn.Module):
    def __init__(self, input_size : int, output_size : int, leaky_alpha : float=0.01):
        super(ResidualBlock, self).__init__()

        self.module = tc.nn.Sequential(
            tc.nn.Conv3d(input_size, output_size, 3, stride=1, padding=1),
            tc.nn.GroupNorm(output_size, output_size),
            tc.nn.LeakyReLU(leaky_alpha, inplace=True),
            tc.nn.Conv3d(output_size, output_size, 3, stride=1, padding=1),
            tc.nn.GroupNorm(output_size, output_size),
            tc.nn.LeakyReLU(leaky_alpha, inplace=True),        
        )

        self.conv = tc.nn.Sequential(
            tc.nn.Conv3d(input_size, output_size, 1)
        )

    def forward(self, x : tc.Tensor):
        return self.module(x) + self.conv(x)







