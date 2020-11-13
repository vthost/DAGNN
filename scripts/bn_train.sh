#!/usr/bin/env bash

set -e

echo $0
echo "Started"
date

source activate dagnn

PROJECT=$PWD
cd dvae

export CUDA_VISIBLE_DEVICES=$1
export PYTHONPATH=$PYTHONPATH:$PROJECT

echo "CUDA_VISIBLE_DEVICES ${CUDA_VISIBLE_DEVICES}"
echo "PYTHONPATH ${PYTHONPATH}"
echo MODEL $2

# TODO SET THIS
RESULTS=../bnresults/
mkdir -p $RESULTS

MODEL=$2
NAME=$MODEL
SAVE=25
BS=128
LR=1e-3
EPOCHS=50

LAYERS=2
AGG_X=0
AGG=attn_h
POOL_ALL=0
POOL=max
DROPOUT=0
BIDIR=1
CLIP=0.25

if [[ "$MODEL" = "DAGNN"* ]]; then
    NAME="${MODEL}_l${LAYERS}_b${BIDIR}_a${AGG}_pa${POOL_ALL}_p${POOL}_c${CLIP}"
fi

python train.py --data-name asia_200k --data-type BN --save-interval $SAVE --lr $LR --save-appendix "_${NAME}" \
                --epochs $EPOCHS --batch-size $BS --model $MODEL  --nz 56 --nvt 8 --res_dir=$RESULTS --keep-old --load-latest-model \
                --dagnn_layers $LAYERS --dagnn_agg_x $AGG_X --dagnn_agg $AGG --bidirectional \
        --dagnn_out_pool_all $POOL_ALL --dagnn_out_pool $POOL --dagnn_dropout $DROPOUT --clip=$CLIP \
        &> $RESULTS/"${NAME}.txt"

echo "Completed"
date
