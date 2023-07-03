import torchio as tio
import torch
import torch.nn as nn

def get_default_device():
    """ Set device """
    if torch.cuda.is_available():
        return torch.device('cuda')
    else:
        return torch.device('cpu')

def preprocessing(x):
    x[x > 2100] = 2100
    x[x < 0] = 0
    return x/2100
    

class PosPadUNet3D(nn.Module):
    def __init__(self, n_classes, emb_shape, in_ch, size=32):
        self.n_classes = n_classes
        self.in_ch = in_ch
        super(PosPadUNet3D, self).__init__()

        self.emb_shape = torch.as_tensor(emb_shape)
        self.pos_emb_layer = nn.Linear(6, torch.prod(self.emb_shape).item())
        self.ec0 = self.conv3Dblock(self.in_ch, size, groups=1)
        self.ec1 = self.conv3Dblock(size, size*2, kernel_size=3, padding=1, groups=1)  # third dimension to even val
        self.ec2 = self.conv3Dblock(size*2, size*2, groups=1)
        self.ec3 = self.conv3Dblock(size*2, size*4, groups=1)
        self.ec4 = self.conv3Dblock(size*4, size*4, groups=1)
        self.ec5 = self.conv3Dblock(size*4, size*8, groups=1)
        self.ec6 = self.conv3Dblock(size*8, size*8, groups=1)
        self.ec7 = self.conv3Dblock(size*8, size*16, groups=1)

        self.pool0 = nn.MaxPool3d(2)
        self.pool1 = nn.MaxPool3d(2)
        self.pool2 = nn.MaxPool3d(2)

        self.dc9 = nn.ConvTranspose3d(size*16+1, size*16, kernel_size=2, stride=2)
        self.dc8 = self.conv3Dblock(size*8 + size*16, size*8, kernel_size=3, stride=1, padding=1)
        self.dc7 = self.conv3Dblock(size*8, size*8, kernel_size=3, stride=1, padding=1)
        self.dc6 = nn.ConvTranspose3d(size*8, size*8, kernel_size=2, stride=2)
        self.dc5 = self.conv3Dblock(size*4 + size*8, size*4, kernel_size=3, stride=1, padding=1)
        self.dc4 = self.conv3Dblock(size*4, size*4, kernel_size=3, stride=1, padding=1)
        self.dc3 = nn.ConvTranspose3d(size*4, size*4, kernel_size=2, stride=2)
        self.dc2 = self.conv3Dblock(size*2 + size*4, size*2, kernel_size=3, stride=1, padding=1)
        self.dc1 = self.conv3Dblock(size*2, size*2, kernel_size=3, stride=1, padding=1)

        self.final = nn.ConvTranspose3d(size*2, n_classes, kernel_size=3, padding=1, stride=1)

        checkpoints = torch.load('checkpoints.pth', map_location=torch.device('cpu'))['state_dict']

        # remove 'module.' as it was originally trained with DataParallel
        from collections import OrderedDict
        new_state_dict = OrderedDict()
        for k, v in checkpoints.items():
            name = k[7:]
            new_state_dict[name] = v
        self.load_state_dict(new_state_dict)

    def conv3Dblock(self, in_channels, out_channels, kernel_size=(3, 3, 3), stride=1, padding=(1, 1, 1), groups=1, padding_mode='replicate'):
        return nn.Sequential(
                nn.Conv3d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, padding_mode=padding_mode, groups=groups),
                nn.BatchNorm3d(out_channels),
                nn.ReLU()
                # nn.SiLU()
        )

    # forward of a single patch
    def patch_forward(self, x, emb_codes):
        h = self.ec0(x)
        feat_0 = self.ec1(h)
        h = self.pool0(feat_0)
        h = self.ec2(h)
        feat_1 = self.ec3(h)

        h = self.pool1(feat_1)
        h = self.ec4(h)
        feat_2 = self.ec5(h)

        h = self.pool2(feat_2)
        h = self.ec6(h)
        h = self.ec7(h)

        emb_pos = self.pos_emb_layer(emb_codes).view(-1, 1, *self.emb_shape)
        h = torch.cat((h, emb_pos), dim=1)
        h = torch.cat((self.dc9(h), feat_2), dim=1)

        h = self.dc8(h)
        h = self.dc7(h)

        h = torch.cat((self.dc6(h), feat_1), dim=1)
        h = self.dc5(h)
        h = self.dc4(h)

        h = torch.cat((self.dc3(h), feat_0), dim=1)
        h = self.dc2(h)
        h = self.dc1(h)
        
        h = self.final(h)
        return torch.sigmoid(h)


    def forward(self, x):
        subject_dict = {
            'volume': tio.ScalarImage(tensor=x),
        }
        subject = tio.Subject(subject_dict)
        grid_sampler = tio.inference.GridSampler(
            subject,
            80,
            0,
        )
        patch_loader = torch.utils.data.DataLoader(grid_sampler, batch_size=1)
        aggregator = tio.inference.GridAggregator(grid_sampler)
        self.eval()

        with torch.inference_mode():
            for patches_batch in patch_loader:
                input_tensor = patches_batch['volume'][tio.DATA].to(get_default_device())
                emb_codes = torch.cat((
                    patches_batch[tio.LOCATION][:,:3],
                    patches_batch[tio.LOCATION][:,:3] + torch.as_tensor(input_tensor.shape[-3:])
                ), dim=1).float().to(get_default_device())

                outputs = self.patch_forward(input_tensor, emb_codes)
                aggregator.add_batch(outputs, patches_batch[tio.LOCATION])

        output_tensor = aggregator.get_output_tensor()
        return output_tensor



if __name__ == '__main__':
    model = PosPadUNet3D(1, [10,10,10], 1)
    x = torch.rand((1, 80, 80, 160))
    model.eval()
    model.to(get_default_device())

    with torch.inference_mode():
        out = model(x)
