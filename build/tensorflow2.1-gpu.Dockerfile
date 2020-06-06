FROM tensorflow/tensorflow:2.1.0-gpu-py3
ARG DEBIAN_FRONTEND=noninteractive

# dependencies
RUN apt-get -y update && apt-get -y install \
    build-essential \
    wget \
    tmux \
    git \
    nano \
    vim \
    libopencv-dev \
    python3-opencv

# python dependencies
RUN pip install --upgrade pip

RUN pip install Cython \
    numpy \
    matplotlib \
    seaborn \
    pandas \
    h5py \
    jupyterlab \
    ipython \
    nose \
    tqdm \
    pyyaml \
    contextlib2

WORKDIR /
RUN git clone https://github.com/ozangungortuhh/yolov3-tf2.git

WORKDIR /yolov3-tf2
RUN pip install -r requirements.txt

COPY ./weights ./weights
RUN python convert.py --weights ./weights/yolov3.weights --output ./checkpoints/yolov3.tf

