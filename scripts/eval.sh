dir=backup
model=faster_rcnn_R_50_FPN_1x_free
network=faster_rcnn_R_50_FPN

gpus=0,1,2,3
gpun=4
master_addr=127.0.0.8
master_port=29508

# suitable for 1080Ti (12G)
# change its value for your GPU
ims_per_batch=$[16*$gpun]

# --- need not change ---
output_dir=$dir/$model
config=config.yml
config_file=$output_dir/$config
vi $config_file

for iter in "0090000"
do
    weight=model_$iter\.pth
    weight_file=$output_dir/$weight
    CUDA_VISIBLE_DEVICES=$gpus python -m torch.distributed.launch --nproc_per_node=$gpun --master_addr $master_addr --master_port $master_port tools/test_net.py --config-file $config_file OUTPUT_DIR $output_dir MODEL.WEIGHT $weight_file TEST.IMS_PER_BATCH $ims_per_batch TEST_ON True NETWORK $network
done
