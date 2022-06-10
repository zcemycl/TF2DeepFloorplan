FROM tensorflow/tensorflow:latest-gpu-py3  

RUN apt-get -y update && apt-get install -y \
        python3-pip \
        software-properties-common \
        wget \
        ffmpeg

COPY requirements.txt /
RUN pip install --upgrade pip setuptools wheel
RUN pip install -r /requirements.txt
RUN gdown https://drive.google.com/uc?id=1czUSFvk6Z49H-zRikTc67g2HUUz4imON
RUN unzip log.zip
RUN rm log.zip
COPY net.py ./net.py
COPY app.py ./app.py
COPY data.py ./data.py
COPY deploy.py ./deploy.py
COPY utils /usr/local/utils
RUN mv /usr/local/utils .
COPY resources /usr/local/resources
RUN mv /usr/local/resources .

CMD ["python","app.py"]

EXPOSE 1111
