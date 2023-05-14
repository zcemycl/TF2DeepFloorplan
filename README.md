# TF2DeepFloorplan [![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](https://www.gnu.org/licenses/gpl-3.0) [<img src="https://colab.research.google.com/assets/colab-badge.svg" >](https://colab.research.google.com/github/zcemycl/TF2DeepFloorplan/blob/master/deepfloorplan.ipynb) ![example workflow](https://github.com/zcemycl/TF2DeepFloorplan/actions/workflows/main.yml/badge.svg) [![Coverage Status](https://coveralls.io/repos/github/zcemycl/TF2DeepFloorplan/badge.svg?branch=main)](https://coveralls.io/github/zcemycl/TF2DeepFloorplan?branch=main)
<!-- [![Hits](https://hits.seeyoufarm.com/api/count/incr/badge.svg?url=https%3A%2F%2Fgithub.com%2Fzcemycl%2FTF2DeepFloorplan&count_bg=%2379C83D&title_bg=%23555555&icon=&icon_color=%23E7E7E7&title=hits&edge_flat=false)](https://hits.seeyoufarm.com) -->
This repo contains a basic procedure to train and deploy the DNN model suggested by the paper ['Deep Floor Plan Recognition using a Multi-task Network with Room-boundary-Guided Attention'](https://arxiv.org/abs/1908.11025). It rewrites the original codes from [zlzeng/DeepFloorplan](https://github.com/zlzeng/DeepFloorplan) into newer versions of Tensorflow and Python.
<br>
Network Architectures from the paper, <br>
<img src="resources/dfpmodel.png" width="50%"><img src="resources/features.png" width="50%">

## Requirements
Install the packages stated in `requirements.txt`, including `matplotlib`,`numpy`,`opencv-python`,`pdbpp`, `tensorflow-gpu` and `tensorboard`. <br>
The code has been tested under the environment of Python 3.7.4 with tensorflow-gpu==2.3.0, cudnn==7.6.5 and cuda10.1_0. Used Nvidia RTX2080-Ti eGPU, 60 epochs take approximately 1 hour to complete.

|OS|Hardware|Application|Command|
|---|---|---|---|
|Ubuntu|CPU|Model Development|`pip install -e .[tfcpu,dev,testing,linting]`|
|Ubuntu|GPU|Model Development|`pip install -e .[tfgpu,dev,testing,linting]`|
|MacOS|M1 Chip|Model Development|`pip install -e .[tfmacm1,dev,testing,linting]`|
|Ubuntu|GPU|Model Deployment API|`pip install -e .[tfgpu,api]`|

## How to run?
1. Install packages via `pip` and `requirements.txt`.
```
python -m venv venv
source venv/bin/activate
pip install --upgrade pip setuptools wheel
pip install -r requirements.txt
```
2. According to the original repo, please download r3d dataset and transform it to tfrecords `r3d.tfrecords`. Friendly reminder: there is another dataset r2v used to train their original repo's model, I did not use it here cos of limited access. Please see the link here [https://github.com/zlzeng/DeepFloorplan/issues/17](https://github.com/zlzeng/DeepFloorplan/issues/17).
3. Run the `train.py` file  to initiate the training, model checkpoint is stored as `log/store/G` and weight is in `model/store`,
```
python dfp/train.py [--batchsize 2][--lr 1e-4][--epochs 1000]
[--logdir 'log/store'][--modeldir 'model/store']
[--saveTensorInterval 10][--saveModelInterval 20]
```
- for example,
```
python dfp/train.py --batchsize=8 --lr=1e-4 --epochs=60
--logdir=log/store --modeldir=model/store
```
4. Run Tensorboard to view the progress of loss and images via,
```
tensorboard --logdir=log/store
```
5. Convert model to tflite via `convert2tflite.py`.
```
python dfp/convert2tflite.py [--modeldir model/store]
[--tflitedir model/store/model.tflite]
[--quantize]
```
6. Download and unzip model from google drive,
```
gdown https://drive.google.com/uc?id=1czUSFvk6Z49H-zRikTc67g2HUUz4imON # log files 112.5mb
unzip log.zip
gdown https://drive.google.com/uc?id=1tuqUPbiZnuubPFHMQqCo1_kFNKq4hU8i # pb files 107.3mb
unzip model.zip
gdown https://drive.google.com/uc?id=1B-Fw-zgufEqiLm00ec2WCMUo5E6RY2eO # tfilte file 37.1mb
unzip tflite.zip
```
7. Deploy the model via `deploy.py`, please be aware that load method parameter should match with weight input.
```
python dfp/deploy.py [--image 'path/to/image']
[--postprocess][--colorize][--save 'path/to/output_image']
[--loadmethod 'log'/'pb'/'tflite']
[--weight 'log/store/G'/'model/store'/'model/store/model.tflite']

```
- for example,
```
python dfp/deploy.py --image floorplan.jpg --weight log/store/G
--postprocess --colorize --save output.jpg --loadmethod log
```

## Docker for API
1. Build and run docker container. (Please train your weight, google drive does not work currently due to its update.)
```
docker build -t tf_docker -f Dockerfile .
docker run -d -p 1111:1111 tf_docker:latest
docker run --gpus all -d -p 1111:1111 tf_docker:latest
```
2. Call the api for output.
```
curl -H "Content-Type: application/json" --request POST  \
  -d '{"uri":"https://cdn.cnn.com/cnnnext/dam/assets/200212132008-04-london-rental-market-intl-exlarge-169.jpg","colorize":1,"postprocess":0, "output":"/tmp"}' \
  http://0.0.0.0:1111/process --output /tmp/tmp.jpg


curl --request POST -F "file=@resources/30939153.jpg;type=image/jpeg" \
  -F "postprocess=0" -F "colorize=0" -F "output=/tmp" http://0.0.0.0:1111/process --output out.jpg
```
3. If you run `app.py` without docker, the second curl for file upload will not work.


## Google Colab
1. Click on [<img src="https://colab.research.google.com/assets/colab-badge.svg" >](https://colab.research.google.com/github/zcemycl/TF2DeepFloorplan/blob/master/deepfloorplan.ipynb) and authorize access.
2. Run the first code cell for installation.
3. Go to Runtime Tab, click on Restart runtime. This ensures the packages installed are enabled.
4. Run the rest of the notebook.

## Deep Floorplan package
1. Install as a package.
```
pip install -e .
python setup.py test
coverage run ./setup.py test
```
2. Import as a package.
```
import dfp
from dfp import net, data
model = net.deepfloorplanModel()
```
3. Uninstall package. `pip uninstall Deep_floorplan`

## How to Contribute?
1. Git clone this repo.
2. Install required packages and pre-commit-hooks.
```
pip install -r requirements.txt
pip install pre-commit
pre-commit install
pre-commit run
pre-commit run --all-files
# pre-commit uninstall/ pip uninstall pre-commit
```
3. Create issues. Maintainer will decide if it requires branch. If so,
```
git fetch origin
git checkout xx-features
```
4. Stage your files, Commit and Push to branch.
5. After pull and merge requests, the issue is solved and the branch is deleted. You can,
```
git checkout main
git pull
git remote prune origin
git branch -d xx-features
```


## Results
- From `train.py` and `tensorboard`.

|Compare Ground Truth (top)<br> against Outputs (bottom)|Total Loss|
|:-------------------------:|:-------------------------:|
|<img src="resources/epoch60.png" width="400">|<img src="resources/Loss.png" width="400">|
|Boundary Loss|Room Loss|
|<img src="resources/LossB.png" width="400">|<img src="resources/LossR.png" width="400">|

- From `deploy.py` and `utils/legend.py`.

|Input|Legend|Output|
|:-------------------------:|:-------------------------:|:-------------------------:|
|<img src="resources/30939153.jpg" width="250">|<img src="resources/legend.png" width="180">|<img src="resources/output.jpg" width="250">|
|`--colorize`|`--postprocess`|`--colorize`<br>`--postprocess`|
|<img src="resources/color.jpg" width="250">|<img src="resources/post.jpg" width="250">|<img src="resources/postcolor.jpg" width="250">|
