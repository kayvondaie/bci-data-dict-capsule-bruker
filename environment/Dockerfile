# hash:sha256:afd50e7142a6b03cbe7d5bd1cd6f6c534498a08cbbd9cbe859a5abdd53562dfb
ARG REGISTRY_HOST
FROM $REGISTRY_HOST/codeocean/jupyterlab:3.4.4-miniconda4.12.0-python3.9-ubuntu20.04

ARG DEBIAN_FRONTEND=noninteractive
ARG GIT_ASKPASS
COPY git-ask-pass /

ENV CODEOCEAN_TOKEN=cop_ZGE5YjljNTc5YTZhNDVhMzg4NjdiNDE2ZjYwYjZjYzVsYUtodXFPYnZFWmdvTzhrdTZVZjhTbmFlOXF0QUtzTTNjODlmMzhj

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        vim=2:8.1.2269-1ubuntu5.32 \
    && rm -rf /var/lib/apt/lists/*

RUN pip install -U --no-cache-dir \
    codeocean==0.15.0 \
    h5py==3.14.0 \
    matplotlib==3.9.4 \
    numpy==2.0.2 \
    opencv-python-headless==4.13.0.92 \
    pandas==2.3.3 \
    scanimage-tiff-reader==1.4.1.4 \
    scipy==1.13.1 \
    tifffile==2024.8.30

COPY postInstall /
RUN --mount=type=secret,id=secrets . /run/secrets/secrets \
    && /postInstall
