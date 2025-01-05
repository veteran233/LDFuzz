LDFuzz
======

This is the official implementation of paper. "A Spatial Semantic Fuzzing Framework for LiDAR-based Autonomous Driving Perception Systems".

This repository contains a fuzzing framework for LiDAR-based termed `LDFuzz`, for testing autonomous driving perception systems.

#### The structure of the repository
```
LDFuzz
├── _assets
├── config
├── data
├── _lib
├── log
├── _others
├── __profile
├── __pycache__
├── utils
├── InitialSeeds.py
├── __init__.py
├── __init__.pyc
├── mlab_visual.py
├── open3d_visual.py
├── README.md
├── requirements.txt
├── rq1step1.py
├── rq1step2.py
├── rq2.py
├── rq3.py
└── run_fuzzer.py
```

## Installation

#### Our working environment
- Ubuntu 20.04
- Python 3.7
- PyTorch 1.13.1
- CUDA 11.7

#### Dependency repository
- [OpenPCDet](https://github.com/open-mmlab/OpenPCDet)

OpenPCDet is a clear, simple, self-contained open source project for LiDAR-based 3D object detection. Please [click here](https://github.com/open-mmlab/OpenPCDet/blob/master/docs/INSTALL.md) for the installation of OpenPCDet.

- [PyGeM](https://github.com/mathLab/PyGeM)

PyGeM (Python Geometrical Morphing) is a python package that allows you to deform a given geometry or mesh with different deformation techniques such as FFD, RBF and IDW. Please [click here](https://github.com/mathLab/PyGeM/blob/master/README.md#dependencies-and-installation) for the installation of PyGeM.

- [LISA](https://github.com/velatkilic/LISA)

LISA is a physics based augmentation method that models effects of adverse weather conditions on lidar data.

Installation of LISA :
```
cd path/to/LISA
python setup.py develop
```

- [MultiTest](https://github.com/MSFTest/MultiTest)

MultiTest employs a physical-aware approach to render modality-consistent object instances using virtual sensors to for Testing Multi-sensor Fusion (MSF) Perception Systems. Please [click here](https://github.com/MSFTest/MultiTest/blob/master/readme.md#installation) for the installation of MultiTest.

Other python package :
```
pip install -r requirements.txt
```

## Usage

#### Dataset
Download KITTI 3D Object Detection Dataset : https://www.cvlibs.net/datasets/kitti/eval_object.php?obj_benchmark=3d

This repository require :
1. velodyne point clouds
2. camera calibration matrices of object data set
3. training labels of object data set

`data` folder tree :
```
data
└── kitti
    ├── ImageSets
    ├── training
    ├── kitti_dataset.yaml
    ├── kitti_dbinfos_train.pkl
    ├── kitti_infos_test.pkl
    ├── kitti_infos_train.pkl
    ├── kitti_infos_trainval.pkl
    └── kitti_infos_val.pkl
```

Extract KITTI training dataset to `data/kitti/training`.

#### Road label

Download road label and extract to `data/kitti/training/road_label`.

#### Prepare for FRD

Running the whole fuzzing process requires downloading the 1024 backbone file for RangeNet++ : [download darknet53-1024.tar.gz](http://www.ipb.uni-bonn.de/html/projects/bonnetal/lidar/semantic/models/darknet53-1024.tar.gz).

`darknet53-1024.tar.gz` file tree :
```
darknet53-1024.tar.gz
├── arch_cfg.yaml
├── backbone
├── data_cfg.yaml
├── segmentation_decoder
└── segmentation_head
```

Extract `backbone` to `_others/lidar_bonnetal/model`.

#### For RQ1
RQ1 seeks to evaluate the effectiveness of transformation operators in identifying erroneous behaviors within LiDAR-based perception systems.

The parameters of `rq1step1.py` :
```
usage: rq1step1.py [-h] -m M [M ...] [--only-seeds]

optional arguments:
  -h, --help    show this help message and exit
  -m M [M ...]  model list, split by blankspace, e.g. pointpillar pv_rcnn
  --only-seeds  if this arg set, it will only generate test seeds
```

1. Generate the seeds that LiDAR-based perception systems needed
```
python rq1step1.py -m pointpillar --only-seeds
```

2. Run RQ1 process, using `-m` argument to select perception systems
```
python rq1step1.py -m pointpillar pv_rcnn second pointrcnn
```

3. Output the result `rq1.pdf`
```
python rq1step2.py
```

#### Before RQ2 / RQ3 / RQ4
This step will implement the process of LDFuzz.

The parameters of `run_fuzzer.py` :
```
usage: run_fuzzer.py [-h] -m M [M ...] -gc GC [GC ...] -s S [S ...] [-mi MI]

optional arguments:
  -h, --help       show this help message and exit
  -m M [M ...]     model list, split by blankspace, e.g. pointpillar pv_rcnn
  -gc GC [GC ...]  guidance criteria list, split by comma, e.g. spc sec
  -s S [S ...]     seed selection strategy list, split by comma, e.g.
                   new,random
  -mi MI           max iteration, default : 1000
```

1. Generate the seeds
```
python InitialSeeds.py
```

2. Run the process of LDFuzz, using `-m` argument to select `perception systems`, using `-gc` argument to select `guidance criteria`, using `-s` argument to select `seed selection strategy`
```
python run_fuzzer.py -m pointpillar pv_rcnn second pointrcnn -gc spc sec ldfuzz none -s new random
```

#### For RQ2
RQ2 independently evaluata the effectiveness of generating test data with and without guidance.

Output the result `rq2_table.xlsx`, `rq2.pdf`.
```
python rq2.py
```

#### For RQ3
RQ3 aims to comprehensively evaluate the effectiveness of LDFuzz in identifying the diversity of erroneous behaviors.

Output the result `rq3_area.pdf`, `rq3_count.pdf`.
```
python rq3.py
```

## Citation
