#!/usr/bin/env bash

set -e

echo $0
echo "Started"
date

## PARAMS
## 1 device
## 2 model

source activate dagnn

PROJECT=$PWD
cd dvae

export CUDA_VISIBLE_DEVICES=$1
export PYTHONPATH=$PYTHONPATH:$PROJECT

echo "CUDA_VISIBLE_DEVICES ${CUDA_VISIBLE_DEVICES}"
echo "PYTHONPATH ${PYTHONPATH}"
echo MODEL $2

# TODO SET THIS
RESULTS=../naresults/
mkdir -p $RESULTS

MODEL=$2
NAME=$MODEL
SAVE=50
BS=32
EPOCHS=100
LR=1e-3

LAYERS=2
AGG=attn_h
POOL_ALL=0
POOL=max
DROPOUT=0
BIDIR=0
CLIP=0.25


if [[ "$MODEL" = "DAGNN"* ]]; then
   NAME="${MODEL}_l${LAYERS}_b${BIDIR}_a${AGG}_pa${POOL_ALL}_p${POOL}_c${CLIP}"
fi

python train.py --data-name final_structures6 --data-type ENAS --save-interval $SAVE --lr $LR --save-appendix "_${NAME}" \
		--epochs $EPOCHS --batch-size $BS --model $MODEL --nz 56 --nvt 6 --res_dir=$RESULTS --keep-old --load-latest-model \
  	--dagnn_layers $LAYERS  --dagnn_agg $AGG  \
  	--dagnn_out_pool_all $POOL_ALL --dagnn_out_pool $POOL --dagnn_dropout $DROPOUT --clip=$CLIP  \
  	&> $RESULTS/"${NAME}.txt"

echo "Completed"
date
