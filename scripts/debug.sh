# faster_rcnn, mask_rcnn, retinanet
det=retinanet
# R_50_FPN, R_101_FPN, X_101_32x8d, ...
backbone=R_50_FPN
# scheduler: 1x, 2x, 
schedule=1x

# try it!
sampling_free_on=False
# 1.0 for retinanet, 2.0 for rcnn
classification_scale=1.0

# gpu env
gpus=0
# number of gpus
gpun=1
# address. keep it different from the running programs.
master_addr=127.0.0.1
master_port=29501

# ------------------------ need not change -----------------------------------
network=$det\_$backbone
config=$network\_$schedule
if [ $sampling_free_on == True ]
then
    dir=backup/$config\_freex$classification_scale
else
    dir=backup/$config\_x0.01
fi
fpn_pos_nms_top_n_train=$[16000/$gpun]

CUDA_VISIBLE_DEVICES=$gpus python -m torch.distributed.launch --nproc_per_node=$gpun --master_addr $master_addr --master_port $master_port tools/train_net.py --config-file configs/$det/$config\.yaml OUTPUT_DIR $dir TEST_ON False MODEL.RPN.FPN_POST_NMS_TOP_N_TRAIN $fpn_pos_nms_top_n_train MODEL.SAMPLING_FREE_ON $sampling_free_on MODEL.CLASSIFICATION_SCALE $classification_scale NETWORK $network
