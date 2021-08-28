# TF2DeepFloorplan [![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](https://www.gnu.org/licenses/gpl-3.0)
This repo contains a basic procedure to train the DNN model suggested by the paper ['Deep Floor Plan Recognition using a Multi-task Network with Room-boundary-Guided Attention'](https://arxiv.org/abs/1908.11025). It rewrites the original codes from [zlzeng/DeepFloorplan](https://github.com/zlzeng/DeepFloorplan) into newer versions of Tensorflow and Python. 
<br>
Network Architectures from the paper, <br>
<img src="resources/dfpmodel.png" width="50%"><img src="resources/features.png" width="50%">

## Requirements
Install the packages stated in `requirements.txt`, including `matplotlib`,`numpy`,`opencv-python`,`pdbpp`, `tensorflow-gpu` and `tensorboard`. <br>
The code has been tested under the environment of Python 3.7.4 with tensorflow-gpu==2.3.0, cudnn==7.6.5 and cuda10.1_0. Used Nvidia RTX2080-Ti eGPU, 60 epochs take approximately 1 hour to complete.

## How to run?
1. Install packages via `pip` and `requirements.txt`.
```
pip install -r requirements.txt
```
2. According to the original repo, please download r3d dataset and transform it to tfrecords `r3d.tfrecords`.
3. Run the `train.py` file  to initiate the training, model weight is stored as `log/store/G`, 
```
python train.py [--batchsize 2][--lr 1e-4][--epochs 1000]
[--logdir 'log/store'][--saveTensorInterval 10][--saveModelInterval 20]
```
- for example,
```
python train.py --batchsize=8 --lr=1e-4 --epochs=60 --logdir=log/store
```
4. Run Tensorboard to view the progress of loss and images via, 
```
tensorboard --logdir=log/store
```
5. Deploy the model via `deploy.py`,
```
python deploy.py [--image 'path/to/image'][--weight 'log/store/G']
[--postprocess][--colorize]
```
- for example,
```
python deploy.py --image floorplan.jpg --weight log/store/G --postprocess --colorize
```

## Results
- From `train.py` and `tensorboard`.

|Compare Ground Truth (top)<br> against Outputs (bottom)|Total Loss|
|:-------------------------:|:-------------------------:|
|<img src="resources/epoch60.png" width="400">|<img src="resources/Loss.png" width="400">|
|Boundary Loss|Room Loss|
|<img src="resources/LossB.png" width="400">|<img src="resources/LossR.png" width="400">|

- From `deploy.py`.

|`--colorize`|`--postprocess`|`--colorize`<br>`--postprocess`|
|:-------------------------:|:-------------------------:|:-------------------------:|
|<img src="resources/color.jpg" width="250">|<img src="resources/post.jpg" width="250">|<img src="resources/postcolor.jpg" width="250">|
