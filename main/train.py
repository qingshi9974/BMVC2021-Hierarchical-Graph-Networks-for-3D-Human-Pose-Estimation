import os
import argparse
import torch
import __init_path
import shutil
import os.path as osp
import numpy as np
from funcs_utils import save_checkpoint, save_plot, check_data_pararell, count_parameters
from core.config import cfg, update_config



parser = argparse.ArgumentParser(description='Train Pose2Mesh')

parser.add_argument('--seed', type=int, default=123, help='random seed to use. Default=123')
parser.add_argument('--resume_training', action='store_true', help='Resume Training')
parser.add_argument('--debug', action='store_true', help='reduce dataset items')
parser.add_argument('--gpu', type=str, default='', help='assign multi-gpus by comma concat')
parser.add_argument('--cfg', type=str, help='experiment configure file name')


args = parser.parse_args()
if args.cfg:
    update_config(args.cfg)
torch.manual_seed(args.seed)
os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)
print("Work on GPU: ", os.environ['CUDA_VISIBLE_DEVICES'])

from core.base import Trainer

trainer = Trainer(args, load_dir='')

print("===> Start training...")
for epoch in range(cfg.TRAIN.begin_epoch, cfg.TRAIN.end_epoch + 1):

    trainer.train(epoch)
    trainer.lr_scheduler.step()
    print(' ===> Start Testing...')
    trainer.test(epoch)


    if epoch > 0:
        is_best = trainer.joint_error < min(trainer.error_history['joint'])
    else:
        is_best = None

    trainer.error_history['joint'].append(trainer.joint_error)


    if is_best:
        save_checkpoint({
            'epoch': epoch,
            'model_state_dict': check_data_pararell(trainer.model.state_dict()),
            'optim_state_dict': trainer.optimizer.state_dict(),
            'scheduler_state_dict': trainer.lr_scheduler.state_dict(),
            'train_log': trainer.loss_history,
            'test_log': trainer.error_history
        }, epoch, is_best)

    save_plot(trainer.loss_history, epoch)
    save_plot(trainer.error_history['joint'], epoch, title='Joint Error')






