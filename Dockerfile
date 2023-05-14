FROM tensorflow/tensorflow:latest-gpu-py3

RUN apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/3bf863cc.pub
RUN apt-get -y update
RUN apt-get install -y python3-pip software-properties-common wget ffmpeg

COPY requirements.txt /
ADD src src
COPY setup.cfg /
COPY setup.py /
COPY pyproject.toml /
RUN pip install --upgrade pip setuptools wheel
WORKDIR /
RUN ls -la
ENV AM_I_IN_A_DOCKER_CONTAINER Yes
RUN pip install opencv-python==4.4.0.44
RUN pip install cmake
RUN pip install -e .[tfgpu,api]
# RUN gdown https://drive.google.com/uc?id=1czUSFvk6Z49H-zRikTc67g2HUUz4imON
# RUN unzip log.zip
# RUN rm log.zip
ADD log log

COPY resources /usr/local/resources
RUN mv /usr/local/resources .

CMD ["python","-m","dfp.app"]

EXPOSE 1111
