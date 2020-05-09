#dir=backup/public
dir=~/.torch/models/maskrcnn_benchmark
det=retinanet
model=retinanet_R_50_FPN_1x
nms=0.5
th=0.05

# gpu and batch
gpus=8,9
gpun=2
master_addr=127.0.0.8
master_port=29508
# suitable for 1080Ti (12G)
ims_per_batch=$[16*$gpun]

# --- need not change ---
output_dir=$dir/$det
config=$model\.yaml
config_file=configs/$det/$config
vi $config_file
weight=$model\.pth
weight_file=$output_dir/$weight

CUDA_VISIBLE_DEVICES=$gpus python -m torch.distributed.launch --nproc_per_node=$gpun --master_addr $master_addr --master_port $master_port tools/test_net.py --config-file $config_file OUTPUT_DIR $output_dir MODEL.WEIGHT $weight_file TEST.IMS_PER_BATCH $ims_per_batch TEST_ON True MODEL.RETINANET.INFERENCE_TH $th MODEL.RETINANET.NMS_TH $nms MODEL.ROI_HEADS.SCORE_THRESH $th MODEL.ROI_HEADS.NMS $nms
