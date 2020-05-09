import torch
from torch import nn
from torch.autograd import Function
from torch.autograd.function import once_differentiable

from maskrcnn_benchmark import _C

# TODO: Use JIT to replace CUDA implementation in the future.
class _AloneBCELoss(Function):
    @staticmethod
    def forward(ctx, logits, targets):
        ctx.save_for_backward(logits, targets)
        num_classes = logits.shape[1]
        ctx.num_classes = num_classes

        losses = _C.alone_bceloss_forward(
            logits, targets, num_classes
        )
        return losses

    @staticmethod
    @once_differentiable
    def backward(ctx, d_loss):
        logits, targets = ctx.saved_tensors
        num_classes = ctx.num_classes
        d_loss = d_loss.contiguous()
        d_logits = _C.alone_bceloss_backward(
            logits, targets, d_loss, num_classes
        )
        return d_logits, None, None, None, None

alone_bce_loss_cuda = _AloneBCELoss.apply


class AloneBCELoss(nn.Module):
    def __init__(self):
        super(AloneBCELoss, self).__init__()

    def forward(self, logits, targets):
        loss = alone_bce_loss_cuda(logits, targets)
        return loss.sum()

    def __repr__(self):
        tmpstr = self.__class__.__name__
        return tmpstr
