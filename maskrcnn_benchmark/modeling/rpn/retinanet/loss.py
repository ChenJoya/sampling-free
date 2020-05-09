"""
This file contains specific functions for computing losses on the RetinaNet
file
"""

import torch
from torch.nn import functional as F

from ..utils import concat_box_prediction_layers

from maskrcnn_benchmark.layers import SigmoidFocalLoss, AloneBCELoss, smooth_l1_loss
from maskrcnn_benchmark.modeling.matcher import Matcher
from maskrcnn_benchmark.modeling.utils import cat
from maskrcnn_benchmark.structures.boxlist_ops import boxlist_iou
from maskrcnn_benchmark.structures.boxlist_ops import cat_boxlist
from maskrcnn_benchmark.modeling.rpn.loss import RPNLossComputationSampling as RPNLossComputation
from maskrcnn_benchmark.modeling.sampling_free import guided_loss_scaling
from torch.distributed import all_reduce

def reduce_sum(tensor):
    tensor = tensor.clone()
    all_reduce(tensor)
    return tensor

class RetinaNetLossComputationFocalloss(RPNLossComputation):
    """
    This class computes the RetinaNet loss.
    """

    def __init__(self, proposal_matcher, box_coder,
                 generate_labels_func,
                 sigmoid_focal_loss,
                 bbox_reg_beta=0.11,
                 regress_norm=1.0):
        """
        Arguments:
            proposal_matcher (Matcher)
            box_coder (BoxCoder)
        """
        self.proposal_matcher = proposal_matcher
        self.box_coder = box_coder
        self.box_cls_loss_func = sigmoid_focal_loss
        self.bbox_reg_beta = bbox_reg_beta
        self.copied_fields = ['labels']
        self.generate_labels_func = generate_labels_func
        self.discard_cases = ['between_thresholds']
        self.regress_norm = regress_norm

    def __call__(self, anchors, box_cls, box_regression, targets):
        """
        Arguments:
            anchors (list[BoxList])
            box_cls (list[Tensor])
            box_regression (list[Tensor])
            targets (list[BoxList])

        Returns:
            retinanet_cls_loss (Tensor)
            retinanet_regression_loss (Tensor
        """
        anchors = [cat_boxlist(anchors_per_image) for anchors_per_image in anchors]
        labels, regression_targets = self.prepare_targets(anchors, targets)

        N = len(labels)
        box_cls, box_regression = \
                concat_box_prediction_layers(box_cls, box_regression)

        labels = torch.cat(labels, dim=0)
        regression_targets = torch.cat(regression_targets, dim=0)
        pos_inds = torch.nonzero(labels > 0).squeeze(1)

        retinanet_regression_loss = smooth_l1_loss(
            box_regression[pos_inds],
            regression_targets[pos_inds],
            beta=self.bbox_reg_beta,
            size_average=False,
        ) / (max(1, pos_inds.numel() * self.regress_norm))

        labels = labels.int()

        retinanet_cls_loss = self.box_cls_loss_func(
            box_cls,
            labels
        ) / (pos_inds.numel() + N)
        
        return retinanet_cls_loss, retinanet_regression_loss

class RetinaNetLossComputationFree(RPNLossComputation):
    """
    This class computes the RetinaNet loss.
    """

    def __init__(self, proposal_matcher, box_coder,
                 generate_labels_func,
                 sigmoid_focal_loss,
                 bbox_reg_beta=0.11,
                 regress_norm=1.0):
        """
        Arguments:
            proposal_matcher (Matcher)
            box_coder (BoxCoder)
        """
        self.proposal_matcher = proposal_matcher
        self.box_coder = box_coder
        self.bbox_reg_beta = bbox_reg_beta
        self.copied_fields = ['labels']
        self.generate_labels_func = generate_labels_func
        self.discard_cases = ['between_thresholds']
        self.regress_norm = regress_norm

    def __call__(self, anchors, box_cls, box_regression, targets):
        """
        Arguments:
            anchors (list[BoxList])
            box_cls (list[Tensor])
            box_regression (list[Tensor])
            targets (list[BoxList])

        Returns:
            retinanet_cls_loss (Tensor)
            retinanet_regression_loss (Tensor
        """
        anchors = [cat_boxlist(anchors_per_image) for anchors_per_image in anchors]
        labels, regression_targets = self.prepare_targets(anchors, targets)

        N = len(labels)
        box_cls, box_regression = \
                concat_box_prediction_layers(box_cls, box_regression)

        labels = torch.cat(labels, dim=0)
        regression_targets = torch.cat(regression_targets, dim=0)
        pos_inds = torch.nonzero(labels > 0).squeeze(1)
        pos_numel = pos_inds.numel()

        retinanet_regression_loss = smooth_l1_loss(
            box_regression[pos_inds],
            regression_targets[pos_inds],
            beta=self.bbox_reg_beta,
            size_average=False,
        ) / max(1, pos_numel * self.regress_norm)

        box_cls_labels = torch.zeros_like(box_cls, device=box_cls.device)
        box_cls_labels[pos_inds, labels[pos_inds].long() - 1] = 1
        retinanet_cls_loss = F.binary_cross_entropy_with_logits(
            box_cls, box_cls_labels, reduction="sum") / (pos_inds.numel() + N)
        retinanet_cls_loss = guided_loss_scaling(retinanet_cls_loss, retinanet_regression_loss)

        return retinanet_cls_loss, retinanet_regression_loss

def generate_retinanet_labels(matched_targets):
    labels_per_image = matched_targets.get_field("labels")
    return labels_per_image

def make_retinanet_loss_evaluator(cfg, box_coder):
    matcher = Matcher(
        cfg.MODEL.RETINANET.FG_IOU_THRESHOLD,
        cfg.MODEL.RETINANET.BG_IOU_THRESHOLD,
        allow_low_quality_matches=True,
    )
    sigmoid_focal_loss = SigmoidFocalLoss(
        cfg.MODEL.RETINANET.LOSS_GAMMA,
        cfg.MODEL.RETINANET.LOSS_ALPHA
    )

    scheme = "free" if cfg.MODEL.SAMPLING_FREE_ON else "focalloss"
    loss_evaluator = eval("RetinaNetLossComputation" + scheme.capitalize())(
        matcher,
        box_coder,
        generate_retinanet_labels,
        sigmoid_focal_loss,
        bbox_reg_beta = cfg.MODEL.RETINANET.BBOX_REG_BETA,
        regress_norm = cfg.MODEL.RETINANET.BBOX_REG_WEIGHT,
    )
    return loss_evaluator
