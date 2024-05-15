from torch import nn 
import torch 
import torch.nn.functional as F

class DiceLoss(nn.Module):
    def __init__(self) -> None:
        super(DiceLoss,self).__init__()
    # def forward(self, input , target): ## onehot
        # input = torch.softmax(input , dim=1)
        # target_onehot = torch.zeros_like(input,dtype = torch.bool)
        # target_onehot.scatter_(1, target.type(torch.int64), 1)
        # smooth = 1e-5
        # assert  target_onehot.shape == input.shape
        # intersect = (target_onehot * input).sum([2,3,4])
        # sum_pred = input.sum([2,3,4])
        # sum_gt = target_onehot.sum([2,3,4])
        # dice = (2* intersect + smooth) / (sum_pred + sum_gt +smooth)
        # # dice = (0.2*dice[:,0] + 0.8*dice[:,1])/2
        # dice = dice.mean(1)
        # dice = dice.sum(0)
        # return 1-dice

    def forward(self, input , target):
        input = torch.softmax(input , dim=1)
        input = input[:,1,:]
        target = target[:,0,:]
        smooth = 1e-5
        assert  target.shape == input.shape
        intersect = torch.sum(input*target)
        sum_gt= torch.sum(target)
        sum_pred = torch.sum(input)
        dice = (2* intersect + smooth) / (sum_pred + sum_gt +smooth)
        
        return 1-dice

            
class focal_DC_loss(nn.Module):
    def __init__(self, soft_dice_kwargs, ce_kwargs, weight_ce=1, weight_dice=1, ignore_label=None,
                 dice_class=SoftDiceLoss):
        super(focal_DC_loss, self).__init__()
        if ignore_label is not None:
            ce_kwargs['ignore_index'] = ignore_label

        self.weight_dice = weight_dice
        self.weight_ce = weight_ce
        self.ignore_label = ignore_label
        self.log_smooth = 1e-10
        self.ce = RobustCrossEntropyLoss(**ce_kwargs)
        self.dc = dice_class(apply_nonlin=softmax_helper_dim1, **soft_dice_kwargs)
        self.gamma = 2
    def forward(self, net_output: torch.Tensor, target:torch.Tensor):
        dc_loss = self.dc(net_output, target, loss_mask  = None)
        preds_softmax = torch.softmax(net_output, 1)
        preds_logsoft = torch.log(preds_softmax + self.log_smooth)

        preds_softmax = preds_softmax.gather(1, target.long())
        preds_logsoft = preds_logsoft.gather(1, target.long())

        ce_loss = -torch.mul(torch.pow((1-preds_softmax), self.gamma), preds_logsoft)

        ce_loss = ce_loss.mean()
        loss = dc_loss + ce_loss
        return loss