# TF2DeepFloorplan
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
3. Run the `train.py` file  to initiate the training, 
```
python main.py [--batchsize 2][--lr 1e-4][--epochs 1000]
[--logdir 'log/store'][--saveTensorInterval 10][--saveModelInterval 20]
```
- for example,
```
python main.py --batchsize=8 --lr=1e-4 --epochs=60 --logdir=log/store
```
4. Run Tensorboard to view the progress of loss and images via,
```
tensorboard --logdir=log/store
```

## Result
The following figure illustrates the result of the training image after 60 epochs, the first row is the ground truth (left:input, middle:boundary, right:room-type), the second row is the generated results. However, the result is not yet postprocessed, so the colors do not represent the classes, the edges are not smooth and the same area does not only show one class. <br>
<img src="resources/epoch60.png" width="40%">
<img src="resources/Loss.png" width="40%">
<img src="resources/LossB.png" width="40%">
<img src="resources/LossR.png" width="40%">
