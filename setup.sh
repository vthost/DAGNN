#!/usr/bin/env bash

CUDA=cu101
CUDA_TK=10.1
TORCH=1.5.0

conda create -n dagnn python=3.7 -y
source activate dagnn
conda install pytorch torchvision cudatoolkit=$CUDA_TK -c pytorch
pip install torch-scatter==latest+${CUDA} -f https://pytorch-geometric.com/whl/torch-${TORCH}.html
pip install torch-sparse==latest+${CUDA} -f https://pytorch-geometric.com/whl/torch-${TORCH}.html
pip install torch-cluster==latest+${CUDA} -f https://pytorch-geometric.com/whl/torch-${TORCH}.html
pip install torch-spline-conv==latest+${CUDA} -f https://pytorch-geometric.com/whl/torch-${TORCH}.html
pip install torch-geometric
pip install -r requirements.txt