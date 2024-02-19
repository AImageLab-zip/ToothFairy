### Ecosystem Imports ###
import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), "."))
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from typing import Iterable

### External Imports ###
import torch as tc
import torch.nn.functional as F
import torchsummary as ts

### Internal Imports ###

import building_blocks as bb

########################


def default_config() -> dict:
    ### Define Params ###
    input_channels = [1, 16, 64, 128, 256]
    output_channels = [16, 64, 128, 256, 512]
    blocks_per_encoder_channel = [1, 1, 2, 2, 2]
    blocks_per_decoder_channel = [1, 1, 2, 2, 2]
    
    ### Parse ###
    config = {}
    config['input_channels'] = input_channels
    config['output_channels'] = output_channels
    config['blocks_per_encoder_channel'] = blocks_per_encoder_channel
    config['blocks_per_decoder_channel'] = blocks_per_decoder_channel
    config['img_size'] = None
    return config

def default_config2() -> dict:
    ### Define Params ###
    input_channels = [1, 10, 32, 64, 128]
    output_channels = [10, 32, 64, 128, 256]
    blocks_per_encoder_channel = [1, 1, 2, 2, 2]
    blocks_per_decoder_channel = [1, 1, 2, 2, 2]
    
    ### Parse ###
    config = {}
    config['input_channels'] = input_channels
    config['output_channels'] = output_channels
    config['blocks_per_encoder_channel'] = blocks_per_encoder_channel
    config['blocks_per_decoder_channel'] = blocks_per_decoder_channel
    config['img_size'] = None
    return config

def low_memory_config() -> dict:
    ### Define Params ###
    input_channels = [1, 6, 16, 64, 128, 256]
    output_channels = [6, 16, 64, 128, 256, 512]
    blocks_per_encoder_channel = [1, 1, 1, 2, 2, 2]
    blocks_per_decoder_channel = [1, 1, 1, 2, 2, 2]
    use_sigmoid = True
    
    ### Parse ###
    config = {}
    config['input_channels'] = input_channels
    config['output_channels'] = output_channels
    config['blocks_per_encoder_channel'] = blocks_per_encoder_channel
    config['blocks_per_decoder_channel'] = blocks_per_decoder_channel
    config['img_size'] = None
    config['use_sigmoid'] = use_sigmoid
    return config

def low_memory_config2() -> dict:
    ### Define Params ###
    input_channels = [1, 6, 16, 64, 128, 256]
    output_channels = [6, 16, 64, 128, 256, 512]
    blocks_per_encoder_channel = [1, 1, 1, 2, 2, 2]
    blocks_per_decoder_channel = [1, 1, 1, 2, 2, 2]
    
    ### Parse ###
    config = {}
    config['input_channels'] = input_channels
    config['output_channels'] = output_channels
    config['blocks_per_encoder_channel'] = blocks_per_encoder_channel
    config['blocks_per_decoder_channel'] = blocks_per_decoder_channel
    config['img_size'] = None
    return config

def low_memory_config3() -> dict:
    ### Define Params ###
    input_channels = [1, 8, 16, 64, 128, 256]
    output_channels = [8, 16, 64, 128, 256, 512]
    blocks_per_encoder_channel = [1, 1, 1, 2, 2, 2]
    blocks_per_decoder_channel = [1, 1, 1, 2, 2, 2]
    use_sigmoid = True
    
    ### Parse ###
    config = {}
    config['input_channels'] = input_channels
    config['output_channels'] = output_channels
    config['blocks_per_encoder_channel'] = blocks_per_encoder_channel
    config['blocks_per_decoder_channel'] = blocks_per_decoder_channel
    config['img_size'] = None
    config['use_sigmoid'] = use_sigmoid
    return config

def larger_config() -> dict:
    ### Define Params ###
    input_channels = [1, 16, 64, 128, 256, 512]
    output_channels = [16, 64, 128, 256, 512, 1024]
    blocks_per_encoder_channel = [1, 1, 2, 2, 2, 2]
    blocks_per_decoder_channel = [1, 1, 2, 2, 2, 2]
    
    ### Parse ###
    config = {}
    config['input_channels'] = input_channels
    config['output_channels'] = output_channels
    config['blocks_per_encoder_channel'] = blocks_per_encoder_channel
    config['blocks_per_decoder_channel'] = blocks_per_decoder_channel
    config['img_size'] = None
    return config


class RUNetEncoder(tc.nn.Module):
    def __init__(self, input_channels : Iterable[int], output_channels : Iterable[int], blocks_per_channel : Iterable[int]):
        super(RUNetEncoder, self).__init__()
        self.input_channels = input_channels
        self.output_channels = output_channels
        self.blocks_per_channel = blocks_per_channel
        self.num_channels = len(input_channels)
        
        if len(self.input_channels) != len(self.output_channels):
            raise ValueError("Number of input channels must be equal to the number of output channels.")
        
        for i, (ic, oc, bpc) in enumerate(zip(input_channels, output_channels, blocks_per_channel)):
            module_list = []
            for j in range(bpc):
                if j == 0:
                    module_list.append(bb.ResidualBlock(ic, oc))
                else:
                    module_list.append(bb.ResidualBlock(oc, oc))
            cic = ic if bpc == 0 else oc
            module_list.append(tc.nn.Conv3d(cic, oc, 4, stride=2, padding=1))
            module_list.append(tc.nn.GroupNorm(oc, oc))
            module_list.append(tc.nn.LeakyReLU(0.01, inplace=True))
            layer = tc.nn.Sequential(*module_list)
            setattr(self, f"encoder_{i}", layer)
        
    def forward(self, x : tc.Tensor) -> Iterable[tc.Tensor]:
        embeddings = []
        cx = x
        for i in range(self.num_channels):
            cx = getattr(self, f"encoder_{i}")(cx)
            embeddings.append(cx)
        return embeddings

class RUNetDecoder(tc.nn.Module):
    def __init__(self, input_channels : Iterable[int], output_channels : Iterable[int], blocks_per_channel : Iterable[int]):
        super(RUNetDecoder, self).__init__()
        self.input_channels = input_channels
        self.output_channels = output_channels
        self.blocks_per_channel = blocks_per_channel
        self.num_channels = len(input_channels)
        
        for i, (ic, oc, bpc) in enumerate(zip(input_channels, output_channels, blocks_per_channel)):
            module_list = []
            coc = oc if i == self.num_channels - 1 else oc + output_channels[i + 1]
            for j in range(bpc):
                if j == 0: 
                    module_list.append(bb.ResidualBlock(coc, oc))
                else:
                    module_list.append(bb.ResidualBlock(oc, oc))
            cic = coc if bpc == 0 else oc
            module_list.append(tc.nn.ConvTranspose3d(cic, oc, 4, stride=2, padding=1))
            module_list.append(tc.nn.GroupNorm(oc, oc))
            module_list.append(tc.nn.LeakyReLU(0.01, inplace=True))
            layer = tc.nn.Sequential(*module_list)
            setattr(self, f"decoder_{i}", layer)       

    def forward(self, embeddings : Iterable[tc.Tensor]) -> tc.Tensor:
        for i in range(self.num_channels - 1, -1, -1):
            if i == self.num_channels - 1:
                cx = getattr(self, f"decoder_{i}")(embeddings[i])         
            else:
                cx = getattr(self, f"decoder_{i}")(tc.cat((bb.pad(cx, embeddings[i]), embeddings[i]), dim=1))       
        return cx
        
class RUNet(tc.nn.Module):
    def __init__(self, input_channels : Iterable[int], output_channels : Iterable[int], blocks_per_encoder_channel : Iterable[int], blocks_per_decoder_channel : Iterable[int], img_size : tuple=None, number_of_output_channels : int=1, use_sigmoid=True):
        super(RUNet, self).__init__()    
        self.input_channels = input_channels
        self.output_channels = output_channels
        self.blocks_per_encoder_channel = blocks_per_encoder_channel
        self.blocks_per_decoder_channel = blocks_per_decoder_channel
        self.number_of_output_channels = number_of_output_channels
        self.image_size = img_size
        self.use_sigmoid = use_sigmoid
        
        if len(self.input_channels) != len(self.output_channels):
            raise ValueError("Number of input channels must be equal to the number of output channels.")
        
        self.encoder = RUNetEncoder(self.input_channels, self.output_channels, self.blocks_per_encoder_channel)
        self.decoder = RUNetDecoder(self.input_channels, self.output_channels, self.blocks_per_decoder_channel)
        if self.use_sigmoid:
            self.last_layer = tc.nn.Sequential(
                tc.nn.Conv3d(in_channels=self.output_channels[0], out_channels=self.number_of_output_channels, kernel_size=1),
                tc.nn.Sigmoid()
            )
        else:
            self.last_layer = tc.nn.Sequential(
                tc.nn.Conv3d(in_channels=self.output_channels[0], out_channels=self.number_of_output_channels, kernel_size=1),
            )            
        
    def forward(self, x : tc.Tensor) -> tc.Tensor:
        _, _, d, h, w = x.shape
        if self.image_size is not None and (d, h, w) != (self.image_size[0], self.image_size[1], self.image_size[2]):
            x = F.interpolate(x, self.image_size, mode='trilinear')
            
        embeddings = self.encoder(x)
        decoded = self.decoder(embeddings)
        if decoded.shape != x.shape:
            decoded = bb.pad(decoded, x)
        result = self.last_layer(decoded)
        
        if self.image_size is not None and (d, h, w) != (self.image_size[0], self.image_size[1], self.image_size[2]):
            result = F.interpolate(result, (d, h, w), mode='trilinear')
        return result
    



### Verification ###

def test_channels_1():
    device = "cuda:0"
    input_channels = [1, 16, 32, 64]
    output_channels = [16, 32, 64, 128]
    blocks_per_encoder_channel = [0, 1, 1, 1]
    blocks_per_decoder_channel = [0, 1, 1, 1]
    model = RUNet(input_channels, output_channels, blocks_per_encoder_channel, blocks_per_decoder_channel).to(device)
    num_samples = 1
    num_channels = 1
    y_size = 256
    x_size = 256
    z_size = 256
    input = tc.randn((num_samples, num_channels, y_size, x_size, z_size), device=device)
    result = model(input)
    print(f"Result shape: {result.shape}")
    ts.summary(model, input_data=input, device=device)  
    
def test_channels_2():
    device = "cuda:0"
    input_channels = [1, 16, 32]
    output_channels = [16, 32, 64]
    blocks_per_encoder_channel = [0, 1, 1]
    blocks_per_decoder_channel = [0, 1, 1]
    model = RUNet(input_channels, output_channels, blocks_per_encoder_channel, blocks_per_decoder_channel).to(device)
    num_samples = 1
    num_channels = 1
    y_size = 128
    x_size = 128
    z_size = 128
    input = tc.randn((num_samples, num_channels, y_size, x_size, z_size), device=device)
    result = model(input)
    print(f"Result shape: {result.shape}")
    ts.summary(model, input_data=input, device=device)  

def test_channels_3():
    device = "cuda:0"
    input_channels = [1, 16, 32, 64, 128]
    output_channels = [16, 32, 64, 128, 128]
    blocks_per_encoder_channel = [0, 1, 1, 1, 1]
    blocks_per_decoder_channel = [0, 1, 1, 1, 1]
    model = RUNet(input_channels, output_channels, blocks_per_encoder_channel, blocks_per_decoder_channel).to(device)
    num_samples = 1
    num_channels = 1
    y_size = 379
    x_size = 318
    z_size = 299
    input = tc.randn((num_samples, num_channels, y_size, x_size, z_size), device=device)
    result = model(input)
    print(f"Result shape: {result.shape}")
    ts.summary(model, input_data=input, device=device)  
    
def test_channels_4():
    device = "cuda:0"
    input_channels = [1, 16, 64, 64, 256, 256]
    output_channels = [16, 64, 64, 256, 256, 512]
    blocks_per_encoder_channel = [1, 1, 1, 1, 1, 2]
    blocks_per_decoder_channel = [1, 1, 1, 1, 1, 2]
    model = RUNet(input_channels, output_channels, blocks_per_encoder_channel, blocks_per_decoder_channel).to(device)
    num_samples = 1
    num_channels = 1
    y_size = 247
    x_size = 197
    z_size = 243
    input = tc.randn((num_samples, num_channels, y_size, x_size, z_size), device=device)
    result = model(input)
    print(f"Result shape: {result.shape}")
    ts.summary(model, input_data=input, device=device)  
    
def test_channels_5():
    device = "cuda:0"
    config = larger_config()
    model = RUNet(**config).to(device)
    num_samples = 1
    num_channels = 1
    y_size = 247
    x_size = 197
    z_size = 243
    input = tc.randn((num_samples, num_channels, y_size, x_size, z_size), device=device)
    result = model(input)
    print(f"Result shape: {result.shape}")
    ts.summary(model, input_data=input, device=device, depth=5)  



def test_channels_6():
    device = "cuda:0"
    config = low_memory_config2()
    model = RUNet(**config).to(device)
    num_samples = 1
    num_channels = 1
    y_size = 400
    x_size = 400
    z_size = 400
    input = tc.randn((num_samples, num_channels, y_size, x_size, z_size), device=device)
    # result = model(input)
    # print(f"Result shape: {result.shape}")
    ts.summary(model, input_data=input, device=device, depth=5)  

def test_channels_7():
    device = "cuda:0"
    config = default_config()
    model = RUNet(**config).to(device)
    num_samples = 1
    num_channels = 1
    y_size = 400
    x_size = 400
    z_size = 400
    input = tc.randn((num_samples, num_channels, y_size, x_size, z_size), device=device)
    # result = model(input)
    # print(f"Result shape: {result.shape}")
    ts.summary(model, input_data=input, device=device, depth=5)  


def run():
    # test_channels_1()
    # test_channels_2()
    # test_channels_3()
    # test_channels_4()
    # test_channels_5()
    # test_channels_6()
    test_channels_7()
    pass

if __name__ == "__main__":
    run()