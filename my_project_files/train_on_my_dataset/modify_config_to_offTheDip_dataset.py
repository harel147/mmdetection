from mmdet.apis import set_random_seed
from mmcv import Config

cfg = Config.fromfile('./configs/faster_rcnn/faster_rcnn_r50_caffe_fpn_mstrain_1x_coco.py')

# Modify dataset type and path
cfg.dataset_type = 'OffTheDipDataset'
cfg.data_root = 'my_project_files/data_training/'

cfg.data.test.type = 'OffTheDipDataset'
cfg.data.test.data_root = 'my_project_files/data_training/'
cfg.data.test.ann_file = 'train.txt'
cfg.data.test.img_prefix = 'labeled_frames/video3'

cfg.data.train.type = 'OffTheDipDataset'
cfg.data.train.data_root = 'my_project_files/data_training/'
cfg.data.train.ann_file = 'train.txt'
cfg.data.train.img_prefix = 'labeled_frames/video3'

cfg.data.val.type = 'OffTheDipDataset'
cfg.data.val.data_root = 'my_project_files/data_training/'
cfg.data.val.ann_file = 'val.txt'
cfg.data.val.img_prefix = 'labeled_frames/video3'

# set model device
cfg.device = 'cuda'
#device='cuda:0'
#cfg.model.to(device)  # Convert the model to GPU
#cfg.model.eval()  # Convert the model into evaluation mode

# modify num classes of the model in box head
cfg.model.roi_head.bbox_head.num_classes = 1
# If we need to finetune a model based on a pre-trained detector, we need to
# use load_from to set the path of checkpoints.
cfg.load_from = 'checkpoints/faster_rcnn_r50_caffe_fpn_mstrain_3x_coco_20210526_095054-1f77628b.pth'

# Set up working dir to save files and logs.
cfg.work_dir = './tutorial_exps'

# The original learning rate (LR) is set for 8-GPU training.
# We divide it by 8 since we only use one GPU.
cfg.optimizer.lr = 0.02 / 8
cfg.lr_config.warmup = None
cfg.log_config.interval = 10

# Change the evaluation metric since we use customized dataset.
cfg.evaluation.metric = 'mAP'
# We can set the evaluation interval to reduce the evaluation times
cfg.evaluation.interval = 12
# We can set the checkpoint saving interval to reduce the storage cost
cfg.checkpoint_config.interval = 12

# Set seed thus the results are more reproducible
cfg.seed = 0
set_random_seed(0, deterministic=False)
cfg.gpu_ids = range(1)

# We can also use tensorboard to log the training process
cfg.log_config.hooks = [
    dict(type='TextLoggerHook'),
    dict(type='TensorboardLoggerHook')]

# We can initialize the logger for training and have a look
# at the final config used for training
print(f'Config:\n{cfg.pretty_text}')

# save the config as a .py file so we could use it later for inference
dump_file = 'tutorial_exps/my_config.py'
cfg.dump(dump_file)