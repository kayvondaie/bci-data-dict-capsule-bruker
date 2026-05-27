# hash:sha256:5da9d34550758dff915b0847bc6c97b86944f412132345dabdeeeb45f71a3f7e
ARG REGISTRY_HOST
FROM $REGISTRY_HOST/codeocean/jupyterlab:3.4.4-miniconda4.12.0-python3.9-ubuntu20.04

ARG DEBIAN_FRONTEND=noninteractive
ARG GIT_ASKPASS
COPY git-ask-pass /

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        vim \
    && rm -rf /var/lib/apt/lists/*

RUN pip install -U --no-cache-dir \
    h5py \
    matplotlib \
    numpy \
    opencv-python-headless \
    pandas \
    scanimage-tiff-reader \
    scipy \
    tifffile
