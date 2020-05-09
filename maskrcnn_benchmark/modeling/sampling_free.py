import torch
from torch import nn
from torch.distributed import all_reduce

from maskrcnn_benchmark.modeling.rpn.anchor_generator import make_anchor_generator, make_anchor_generator_retinanet
from maskrcnn_benchmark.structures.boxlist_ops import boxlist_iou
from maskrcnn_benchmark.modeling.matcher import Matcher
from maskrcnn_benchmark.data import make_init_data_loader
from maskrcnn_benchmark.structures.image_list import to_image_list
from maskrcnn_benchmark.modeling.backbone import build_backbone
from maskrcnn_benchmark.structures.boxlist_ops import cat_boxlist

from math import ceil, log
from tqdm import tqdm
import os
import json

def get_num_gpus():
    return int(os.environ["WORLD_SIZE"]) if "WORLD_SIZE" in os.environ else 1

def reduce_div(tensor1, tensor2):
    tensor1 = tensor1.clone()
    tensor2 = tensor2.clone()
    all_reduce(tensor1)
    all_reduce(tensor2)
    return tensor1 / tensor2

def request_arch(cfg):
    return cfg.NETWORK

def load_prior(cfg, arch, filename="init_prior.json"):
    if not os.path.exists(filename):
        print("Initialization file %s is not existed. Calculating prior from model (1 epoch)."%(filename))
        return None 
    model = request_arch(cfg)
    print('Find prior of model %s'%(model))
    with open(filename, 'r') as f:
        model_prior_dict = json.load(f) 
        if model in model_prior_dict:
            print('Find it. Use it to initialize the model.')
            return model_prior_dict[model] 
        else:
            print('Not find. Calculating prior from model (1 epoch).')
            return None 

def save_prior(cfg, prior, arch, filename="init_prior.json"):
    model = request_arch(cfg)

    if os.path.exists(filename):
        with open(filename, 'r') as f:
            model_prior_dict = json.load(f)
            model_prior_dict[model] = prior
    else:
        print("Initialization file %s is not existed. Create it."%(filename))
        model_prior_dict = {model : prior}
        
    with open(filename, 'w') as f:
        json.dump(model_prior_dict, f)

    print("Priors have saved to %s."%(filename), prior)

def optimal_bias_init(cfg, bias):
    device = torch.device(cfg.MODEL.DEVICE)
    arch = request_arch(cfg)
    if 'retinanet' in arch: 
        anchor_generator = make_anchor_generator_retinanet(cfg)
        fg_iou, bg_iou = cfg.MODEL.RETINANET.FG_IOU_THRESHOLD, cfg.MODEL.RETINANET.BG_IOU_THRESHOLD
        num_classes = cfg.MODEL.RETINANET.NUM_CLASSES - 1
        num_anchors = len(cfg.MODEL.RETINANET.ASPECT_RATIOS) \
            * cfg.MODEL.RETINANET.SCALES_PER_OCTAVE
    else:
        assert 'faster_rcnn' in arch
        anchor_generator = make_anchor_generator(cfg)
        fg_iou, bg_iou = cfg.MODEL.RPN.FG_IOU_THRESHOLD, cfg.MODEL.RPN.BG_IOU_THRESHOLD
        num_classes = 1
        num_anchors = anchor_generator.num_anchors_per_location()[0] 
    
    prior = load_prior(cfg, arch)

    if prior is not None:
        nn.init.constant_(bias, -log((1 - prior) / prior))            
        return

    data_loader = make_init_data_loader(
        cfg, is_distributed=True, images_per_batch=cfg.SOLVER.IMS_PER_BATCH
    )

    proposal_matcher = Matcher(
        fg_iou,
        bg_iou,
        allow_low_quality_matches=True,
    )

    backbone = build_backbone(cfg).to(device)
    num_fg, num_all = 0, 0
    num_gpus = get_num_gpus()
    
    for images, targets, _ in tqdm(data_loader):
        images = images.to(device)
        targets = [target.to(device) for target in targets]
        h, w = images.tensors.shape[-2:]

        if num_all == 0:
            features = backbone(images.tensors)
            n, c = features[0].shape[:2]
            levels = len(features)
            stride = int(h / features[0].shape[2])
    
        features = [torch.zeros(n, c, int(ceil(h / (stride * 2 ** i))), int(ceil(w / (stride * 2 ** i))), device=device) for i in range(levels)]
        anchors = anchor_generator(images, features)
        anchors = [cat_boxlist(anchors_per_image).to(device) for anchors_per_image in anchors]
            
        for anchor, target in zip(anchors, targets):
            match_quality_matrix = boxlist_iou(target, anchor)
            matched_idxs = proposal_matcher(match_quality_matrix)
            num_fg_per_image, num_bg_per_image = (matched_idxs >= 0).sum(), (matched_idxs == Matcher.BELOW_LOW_THRESHOLD).sum()
            num_fg += num_fg_per_image
            num_all += num_fg_per_image + num_bg_per_image
    fg_all_ratio = reduce_div(num_fg.float(), num_all.float()).item()
    prior = fg_all_ratio / num_classes
    nn.init.constant_(bias, -log((1 - prior) / prior))
    if torch.cuda.current_device() == 0:
        save_prior(cfg, prior, arch)

def guided_loss_scaling(box_cls_loss, box_reg_loss):
    with torch.no_grad():
        w = box_reg_loss / box_cls_loss
    box_cls_loss *= w
    return box_cls_loss

def adaptive_threshold(cfg, num_classes):
    arch = request_arch(cfg) 
    return load_prior(cfg, arch) * num_classes
