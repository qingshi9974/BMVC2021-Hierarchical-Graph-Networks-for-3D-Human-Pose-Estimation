from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
import time
import os.path as osp
import numpy as np
import cv2
import math
import torch
import scipy.io as io
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from collections import Counter
import argparse
import Human36M.dataset

from models.hgn import HGN
from multiple_datasets import MultipleDatasets
from core.loss import get_loss
from core.config import cfg
from display_utils import display_model
from funcs_utils import get_optimizer, load_checkpoint, get_scheduler, count_parameters, stop, lr_check, save_obj,AverageMeter
from vis import vis_2d_pose, vis_3d_pose

def get_dataloader(args, dataset_names, is_train):
    dataset_split = 'TRAIN' if is_train else 'TEST'
    batch_per_dataset = cfg[dataset_split].batch_size // len(dataset_names)
    dataset_list, dataloader_list = [], []

    print(f"==> Preparing {dataset_split} Dataloader...")
    for name in dataset_names:
        dataset = eval(f'{name}.dataset')(dataset_split.lower(), args=args)
        print("# of {} {} data: {}".format(dataset_split, name, len(dataset)))
        print(dataset)
        dataloader = DataLoader(dataset,
                                batch_size=batch_per_dataset,
                                shuffle=cfg[dataset_split].shuffle,
                                num_workers=cfg.DATASET.workers,
                                pin_memory=False)
        dataset_list.append(dataset)
        dataloader_list.append(dataloader)

    if not is_train:
        return dataset_list, dataloader_list
    else:
        trainset_loader = MultipleDatasets(dataset_list, make_same_len=True)
        batch_generator = DataLoader(dataset=trainset_loader, batch_size=batch_per_dataset * len(dataset_names), shuffle=cfg[dataset_split].shuffle,
                                     num_workers=cfg.DATASET.workers, pin_memory=False)
        return dataset_list, batch_generator


def prepare_network(args, load_dir='', is_train=True):
    dataset_names = cfg.DATASET.train_list if is_train else cfg.DATASET.test_list
    dataset_list, dataloader = get_dataloader(args, dataset_names, is_train)
    
    model, criterion, optimizer, lr_scheduler = None, None, None, None
    loss_history, test_error_history = [], {'surface': [], 'joint': []}

    main_dataset = dataset_list[0]
    if is_train or load_dir:
        print(f"==> Preparing {cfg.MODEL.name} MODEL...")
    
        model = HGN(main_dataset.graph_Adj[-1],main_dataset.graph_Adj[-2],main_dataset.graph_Adj[-3],128 ).cuda()
        print('# of model parameters: {}'.format(count_parameters(model)))

    if is_train:
        criterion = get_loss(faces=main_dataset.mesh_model.face)
        optimizer = get_optimizer(model=model)
        lr_scheduler = get_scheduler(optimizer=optimizer)

    if load_dir and (not is_train ):
        print('==> Loading checkpoint')
        checkpoint = load_checkpoint(load_dir=load_dir)
        model.load_state_dict(checkpoint['model_state_dict'])

        if is_train:
            optimizer.load_state_dict(checkpoint['optim_state_dict'])
            for state in optimizer.state.values():
                for k, v in state.items():
                    if torch.is_tensor(v):
                        state[k] = v.cuda()
            curr_lr = 0.0

            for param_group in optimizer.param_groups:
                curr_lr = param_group['lr']

            lr_state = checkpoint['scheduler_state_dict']

            lr_state['milestones'], lr_state['gamma'] = Counter(cfg.TRAIN.lr_step), cfg.TRAIN.lr_factor
            lr_scheduler.load_state_dict(lr_state)

            loss_history = checkpoint['train_log']
            test_error_history = checkpoint['test_log']
            cfg.TRAIN.begin_epoch = checkpoint['epoch'] + 1
            print('===> resume from epoch {:d}, current lr: {:.0e}, milestones: {}, lr factor: {:.0e}'
                  .format(cfg.TRAIN.begin_epoch, curr_lr, lr_state['milestones'], lr_state['gamma']))

    return dataloader, dataset_list, model, criterion, optimizer, lr_scheduler, loss_history, test_error_history

from collections import OrderedDict
class Trainer:
    def __init__(self, args, load_dir,is_test=False):
      
        self.batch_generator, self.dataset_list, self.model, self.loss, self.optimizer, self.lr_scheduler, self.loss_history, self.error_history\
            = prepare_network(args, load_dir=load_dir, is_train=not is_test)
        self.val_loader, self.val_dataset,_, _, _, _, _, _ = \
            prepare_network(args, load_dir=load_dir, is_train=False)
            

        self.main_dataset = self.dataset_list[0]
        self.val_loader, self.val_dataset = self.val_loader[0], self.val_dataset[0]
        self.print_freq = 100

        self.J_regressor = eval(f'torch.Tensor(self.main_dataset.joint_regressor_{cfg.DATASET.target_joint_set}).cuda()')


        self.model = self.model.cuda()
        self.model = nn.DataParallel(self.model)

        self.normal_weight = cfg.MODEL.normal_loss_weight
        self.edge_weight = cfg.MODEL.edge_loss_weight
        self.joint_weight = cfg.MODEL.joint_loss_weight
        self.edge_add_epoch = cfg.TRAIN.edge_loss_start
        self.loss_func = torch.nn.MSELoss()
    def train(self, epoch):
        self.model.train()

        lr_check(self.optimizer, epoch)
       
        running_loss = 0.0
        joint_error = 0 
     
        batch_time = AverageMeter()
        data_time = AverageMeter()
        epoch_loss_3d_pos = AverageMeter()
        epoch_mpjpe_3d_pos = AverageMeter()

        epoch_loss48_3d_pos = AverageMeter()
        epoch_mpjpe48_3d_pos = AverageMeter()

        epoch_loss96_3d_pos = AverageMeter()
        epoch_mpjpe96_3d_pos = AverageMeter()
        end = time.time()
        for i, (inputs, targets,mesh_valid_48,mesh_valid_96) in enumerate(self.batch_generator):
            # convert to cuda
            mesh_valid_48 = mesh_valid_48.cuda()
            mesh_valid_96 = mesh_valid_96.cuda()
            input_pose = inputs['pose2d'].cuda()

            gt_reg3dpose, gt_mesh48,gt_mesh96 = targets['reg_pose3d'].cuda(), targets['mesh_48'].cuda(), targets['mesh_96'].cuda()

            data_time.update(time.time() - end)

            # model
            pred_pose,pred_mesh48,pred_mesh96= self.model(input_pose)   # B x 12288 x 3
            pred_mesh48 = pred_mesh48[:, self.main_dataset.graph_perm_reverse48[:45], :3]
            pred_mesh96 = pred_mesh96[:, self.main_dataset.graph_perm_reverse[:83], :3]
            # pred_mesh = gt_mesh
            gt_reg3dpose = gt_reg3dpose/1000


            j_error = self.main_dataset.compute_joint_err(pred_pose*1000, gt_reg3dpose*1000)
            s_error48 = self.main_dataset.compute_mesh_err(pred_mesh48*1000, gt_mesh48*1000)
            s_error96 = self.main_dataset.compute_mesh_err(pred_mesh96*1000, gt_mesh96*1000)
    
            joint_error += j_error
    
  
            loss4 = self.loss[3](pred_pose,  gt_reg3dpose)
            loss3 = self.loss[4](pred_mesh48,  gt_mesh48.float(),mesh_valid_48)
            loss2 = self.loss[4](pred_mesh96,  gt_mesh96.float(),mesh_valid_96)
         

            num_poses = input_pose.shape[0]
 
            loss =  (loss2 + loss3) + loss4*100
    
            epoch_loss_3d_pos.update(loss4.item(), num_poses)
            epoch_mpjpe_3d_pos.update(j_error,num_poses)

            epoch_loss48_3d_pos.update(loss3.item(), num_poses)
            epoch_mpjpe48_3d_pos.update( s_error48 ,num_poses)

            epoch_loss96_3d_pos.update(loss2.item(), num_poses)
            epoch_mpjpe96_3d_pos.update( s_error96,num_poses)
            
            

            self.optimizer.zero_grad()
            
            loss.backward()

            nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1)
            self.optimizer.step()
            batch_time.update(time.time() - end)
            end = time.time()
          
 

            # log
            freq = self.print_freq
            running_loss += float(loss4.detach().item())
            if i % freq == 0:
                loss4 = loss4.detach()
       
                print(time.strftime("%Y-%m-%d-%H_%M_%S", time.localtime()),
                                                f'Epoch{epoch}_({i}/{len(self.batch_generator)}) => '
                                                f'data:{data_time.avg:4f}  ' 
                                                f'batch:{batch_time.avg:4f}  '
                                               f'loss: {epoch_loss_3d_pos.avg:.4f}   '
                                               f'loss48: {epoch_loss48_3d_pos.avg:.4f}   '
                                               f'loss96: {epoch_loss96_3d_pos.avg:.4f}   '
                                               f' MPJPE: {epoch_mpjpe_3d_pos.avg:.2f}  '
                                               f' 48: {epoch_mpjpe48_3d_pos.avg:.2f}  '
                                               f' 96: {epoch_mpjpe96_3d_pos.avg:.2f}  ')

                joint_error = 0
                accs = 0

        self.loss_history.append(running_loss / len(self.batch_generator))

        print(f'Epoch{epoch} Loss: {self.loss_history[-1]:.4f}')



    def test(self,epoch):

        self.model.eval()     
        joint_error = 0.0
        accs = 0
        result = []
        eval_prefix = f'Epoch{epoch} ' if epoch else ''

        with torch.no_grad():
            for i, (inputs, targets,_,_) in enumerate(self.val_loader):
   
         
                input_pose, gt_pose3d= inputs['pose2d'].cuda(), targets['reg_pose3d'].cuda()

                pred_pose,_,_= self.model(input_pose[:,:,:2]) # B x 12288 x 3
                if input_pose.shape[2]==3:
                     mask = input_pose[:,:,2]
                else:
                    mask = torch.ones(input_pose.shape[0],17).cuda()
         
                j_error = self.val_dataset.compute_joint_err(pred_pose*1000, gt_pose3d) 
                if i % 10 == 0:
                    print(time.strftime("%Y-%m-%d-%H_%M_%S", time.localtime()),
                        f'{eval_prefix}({i}/{len(self.val_loader)})  joint error: {j_error:.4f}')
                    vis_3d_pose(pred_pose[0].detach().cpu().numpy(), self.val_dataset.skeleton, joint_set_name='smpl')
                    vis_3d_pose(gt_pose3d[0].detach().cpu().numpy(), self.val_dataset.skeleton, joint_set_name='smpl')
                joint_error += j_error
                
                # Final Evaluation
                if (epoch % 1 ==0 or epoch == cfg.TRAIN.end_epoch):
                    pred_pose, gt_pose3d ,mask,input_pose= pred_pose.detach().cpu().numpy()*1000, gt_pose3d.detach().cpu().numpy(),mask.detach().cpu().numpy(),input_pose[:,:,:2].detach().cpu().numpy()
                    for j in range(len(input_pose)):
                        out = {}
                        out['joint_coord'], out['joint_cam'],out['mask'] ,out['input']= pred_pose[j], gt_pose3d[j],mask[j],input_pose[j]
                        result.append(out)
            self.joint_error = joint_error / len(self.val_loader)
            accs = accs/len(self.val_loader)
            if (epoch %1 == 0 or epoch == cfg.TRAIN.end_epoch):
                self.val_dataset.evaluate_joint(result)



