# HGN: Hierarchical Graph Networks for 3D Human Pose Estimation

## Introduction
This repository is the offical [Pytorch](https://pytorch.org/) implementation of [Hierarchical Graph Networks for 3D Human Pose Estimation (BMVC 2021)](https://arxiv.org/abs/2111.11927).

## Install guidelines
- We recommend you to use an [Anaconda](https://www.anaconda.com/) virtual environment. Install [PyTorch](https://pytorch.org/) >= 1.2 according to your GPU driver and Python >= 3.7.2, and run `sh requirements.sh`. 

## Directory

### Root

The `${ROOT}` is described as below.

```
${ROOT} 
|-- data
|-- lib
|-- experiment
|-- main
|-- smplpytorch
```
- `data` contains data loading codes and soft links to images and annotations directories.
- `lib` contains kernel codes for HGN.
- `main` contains high-level codes for training or testing the network.
- `experiment` contains the outputs of the system, whic include train logs, trained model weights, and visualized outputs.
- `smplpytorch` contains contains kernel codes for SMPL model .

The `data` directory structure should follow the below hierarchy.
```
${ROOT}  
|-- data  
|   |-- Human36M  
|   |   |-- annotations   
|   |   |   | -- Human36M_subject1_camera.json
|   |   |   | -- Human36M_subject1_data.json
|   |   |   | -- Human36M_subject1_joint_3d.json
|   |   |   | -- Human36M_subject1_smpl_camera.json
|   |   |   | -- ....
|   |   |   | -- Human36M_subject13_smpl_camera.json
|   |   |-- J_regressor_h36m_correct.npy
```

- Download the annotations file (SMPL parameters from SMPLify-X, Human36M joints, Camera parameters) [[Annotations](https://jbox.sjtu.edu.cn/v/link/view/4266d27ff78c45a3b1f7d73a27258e65)], and unzip annotations file to corresponding directory.
- All annotation files follow [MS COCO format](https://cocodataset.org/#format-data).

### Experiment

The `experiment` directory will be created as below.
```
${ROOT}  
|-- experiment  
|   |-- exp_*  
|   |   |-- checkpoint  
|   |   |-- graph 
|   |   |-- vis 
```

- `experiment` contains train/test results of Pose2Mesh on various benchmark datasets.
We recommed you to create the folder as a soft link to a directory with large storage capacity.

- `exp_*` is created for each train/test command. 
The wildcard symbol refers to the time of the experiment train/test started.
Default timezone is UTC+9, but you can set to your local time.

- `checkpoint` contains the model checkpoints for each epoch. 

- `graph` contains visualized train logs of error and loss. 

### Pretrained model weights
The pretrained model weights corresponding to the best model when Ground-Truth 2D keypoints are used as input, which can achieve the best results in MPJPE (37.32mm).
```
${ROOT}  
|-- pretrained
|   |-- final.pth.tar
```

## Running Pose2Mesh

### Start
### Train

you can run the 
```
python main/train.py --gpu 0,1 --cfg ./asset/yaml/HGN_human36J_train_human36.yml
```
### Test
Select the config file in ${ROOT}/asset/yaml/ and test. You can change the pretrained model weight, the default path of pretrained model weight is ${ROOT}/pretrained/'. To save sampled outputs to obj files, change TEST.vis value to True in the config file.
Run
```
python main/test.py --gpu 0,1 --cfg ./asset/yaml/HGN_human36J_test_human36.yml
```
