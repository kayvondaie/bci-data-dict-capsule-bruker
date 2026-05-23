FROM registry.codeocean.allenneuraldynamics.org/codeocean/jupyterlab:3.4.4-miniconda4.12.0-python3.9-ubuntu20.04

ARG DEBIAN_FRONTEND=noninteractive
ENV CAPSULE_VERSION="0.1.0"
ARG GIT_ASKPASS
ARG GIT_ACCESS_TOKEN
COPY git-askpass /

RUN ["apt-get", "update"]
RUN ["apt-get", "install", "-y", "vim"]

RUN curl -fsSL https://code-server.dev/install.sh | sh

RUN pip3 install -U --no-cache-dir \
    numpy \
    scipy \
    pandas \
    h5py \
    matplotlib \
    scanimage-tiff-reader \
    tifffile \
    opencv-python-headless
    