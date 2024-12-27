LDFuzz
======

This is the official implementation of paper. "A Spatial Semantic Fuzzing Framework for LiDAR-based Autonomous Driving Perception Systems".

This repository contains a fuzzing framework for LiDAR-based termed `LDFuzz`, for testing autonomous driving perception systems.


## Installation

Our working environment listed below :
- Ubuntu 20.04
- Python 3.7
- PyTorch 1.13.1
- CUDA 11.7

Dependency repository :
- OpenPCDet : https://github.com/open-mmlab/OpenPCDet
- PyGeM : https://github.com/mathLab/PyGeM
- LISA : https://github.com/velatkilic/LISA
- MultiTest : https://github.com/MSFTest/MultiTest

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

#### For RQ1
1. `python rq1step1.py -m pointpillar --only-seeds`
2. `python rq1step1.py -m pointpillar pv_rcnn second pointrcnn`
3. run `python rq1step1.py` to output PDF format file

#### Before RQ2 / RQ3 / RQ4
1. `python InitialSeeds.py`
2. `python run_fuzzer.py -m pointpillar pv_rcnn second pointrcnn -gc spc sec ldfuzz none -s new random`

#### For RQ2
`python rq2.py`

#### For RQ3
`python rq3.py`

## Citation
