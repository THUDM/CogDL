FROM ubuntu:latest

ARG CUDA=cpu
ARG TORCH=1.7.0

RUN echo BUILDING WITH CUDA===${CUDA} AND TORCH===${TORCH}

RUN apt update
RUN apt upgrade -y
RUN apt install python3 python3-pip git -y
RUN python3 -m pip install torch==${TORCH}+${CUDA} torchvision==0.8.1+${CUDA} torchaudio===0.7.0 -f https://download.pytorch.org/whl/torch_stable.html dgl
RUN python3 -m pip install torch-scatter==latest+${CUDA} -f https://pytorch-geometric.com/whl/torch-${TORCH}.html
RUN python3 -m pip install torch-scatter==latest+${CUDA} -f https://pytorch-geometric.com/whl/torch-${TORCH}.html
RUN python3 -m pip install torch-sparse==latest+${CUDA} -f https://pytorch-geometric.com/whl/torch-${TORCH}.html
RUN python3 -m pip install torch-cluster==latest+${CUDA} -f https://pytorch-geometric.com/whl/torch-${TORCH}.html
RUN python3 -m pip install torch-spline-conv==latest+${CUDA} -f https://pytorch-geometric.com/whl/torch-${TORCH}.html
RUN python3 -m pip install torch-geometric



SHELL ["/bin/bash", "-c"]
