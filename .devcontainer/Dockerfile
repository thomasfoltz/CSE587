FROM nvidia/cuda:12.6.3-cudnn-devel-ubuntu24.04

WORKDIR /workspaces

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y \
    python3.12 \
    python3-venv \
    git \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .

RUN python3 -m venv /workspaces/venv

RUN /workspaces/venv/bin/pip install --no-cache-dir -r requirements.txt

ENV PATH="/workspaces/venv/bin:$PATH"
ENV NVIDIA_VISIBLE_DEVICES=all
ENV NVIDIA_DRIVER_CAPABILITIES=compute,utility

CMD ["bash"]