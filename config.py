batch_size: 32     #24
use_gpu: True
img_size: 256
epoches: 50      #800
base_lr: 1e-3
weight_decay: 2e-5
momentum: 0.9
power: 0.99
gpu_id: '0'
loss_type: 'ce'
save_iter: 2
num_workers: 0
val_visual: True
image_driver: 'gdal'
# color_table: 0,0,0,255,0,0,0,255,0,0,0,255
num_class: 2
thread: 0.4

model_name: 'Res_UNet_50'  # ' unet, res_unet_psp, res_unet' ？？？

pretrained_model: 'None'
extra_loss: False
model_experision: 'v2'





train_list: 'list/train_list.txt'
val_list: 'list/val_list.txt'
test_list: 'list/test_list.txt'
# train_gt: 'dataset\label_data\label_train'
# val_gt: 'dataset\label_data\label_train'

#data path

# dataset: 'massroad'
# exp_name: '1109'

# save_path: 'test_result'