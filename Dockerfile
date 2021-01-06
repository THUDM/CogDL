FROM ubuntu:latest

ARG CUDA=cpu
ARG TORCH=1.7.0

RUN echo BUILDING WITH CUDA===${CUDA} AND TORCH===${TORCH}

RUN apt update
RUN apt upgrade -y
RUN apt install python3 python3-pip git -y
RUN python3 -m pip install torch==${TORCH}+${CUDA} -f https://download.pytorch.org/whl/torch_stable.html
RUN if [ "$CUDA" = "cpu" ]; then python3 -m pip install dgl==0.4.3 ; else python3 -m pip install dgl-${CUDA}==0.4.3 ; fi
RUN python3 -m pip install torch-scatter==latest+${CUDA} -f https://pytorch-geometric.com/whl/torch-${TORCH}.html
RUN python3 -m pip install torch-sparse==latest+${CUDA} -f https://pytorch-geometric.com/whl/torch-${TORCH}.html
RUN python3 -m pip install torch-cluster==latest+${CUDA} -f https://pytorch-geometric.com/whl/torch-${TORCH}.html
RUN python3 -m pip install torch-spline-conv==latest+${CUDA} -f https://pytorch-geometric.com/whl/torch-${TORCH}.html
RUN python3 -m pip install torch-geometric


SHELL ["/bin/bash", "-c"]
