from networks.generic_UNet import initialize_nnunet
from networks.generic_UNetv2 import initialize_nnunetv2
from networks.generic_UNet_small import initialize_nnunet_small

def net_factory_3d(net_type="unet_3D", in_chns=1, class_num=2):
    if net_type == "nnUNet":
        net = initialize_nnunet(num_classes=class_num).cuda()
        # net = initialize_network(num_classes=class_num).cuda()
    elif net_type == "nnUNetv2":
        net = initialize_nnunetv2(num_classes=class_num).cuda()
    elif net_type == "nnUNet_small":
        net = initialize_nnunet_small(num_classes=class_num).cuda()
    else:
        net = None
    return net
