#!/usr/bin/env bash

set -e

source activate dagnn

PROJECT=$PWD

# you can adapt these
DATA=$PROJECT/data
SAVE=$PROJECT/saved_models

MODEL=dagnn

LR=1e-3
CLIP=0.25
BATCHSIZE=160
TIDX=train15  # use "" to run over the full dataset
FOLDS=5
EPOCHS=1000
LAY=5
PAT=10

EA=1
LAYDAG=2
BIDIR=1
AGG_X=0
AGG=attn_h
POOL_ALL=0
POOL=max
DROPOUT=0
MAPPER_BIAS=1

export CUDA_VISIBLE_DEVICES=0,1
export PYTHONPATH=$PYTHONPATH:$PROJECT

echo "PYTHONPATH: ${PYTHONPATH}"
echo started experiments

cd ogbg-code


NAME="${MODEL}"
if [[ "$MODEL" = "dagnn" ]]; then
    NAME="dagnn_ea${EA}_l${LAYDAG}_b${BIDIR}_ax${AGG_X}_a${AGG}_pa${POOL_ALL}_p${POOL}_c${CLIP}"
fi
NAME="${NAME}_lr${LR}"
if [[ "$TIDX" != "" ]]; then
    NAME="${NAME}_${TIDX}"
fi

CHECKPOINT=""

for f in $SAVE/$NAME*; do
    if [ -e "$f" ]; then
      CHECKPOINT=`basename $f`
      echo "checkpoint exists! $f"
      echo "using: $CHECKPOINT"
      break
    fi
done

python main_pyg.py --gnn=$MODEL --drop_ratio=0 --max_seq_len=5 --num_vocab=5000 --lr=$LR \
      --num_layer=$LAY  --emb_dim=300 --batch_size=$BATCHSIZE --folds=$FOLDS --epochs=$EPOCHS --num_workers=$BATCHSIZE --dataset="ogbg-code" \
      --dir_data=$DATA --dir_save=$SAVE --filename=$NAME --train_idx=$TIDX --clip=$CLIP --dagnn_mapper_bias=$MAPPER_BIAS \
      --dagnn_wea=$EA --dagnn_layers=$LAYDAG --dagnn_bidir=$BIDIR --dagnn_agg_x=$AGG_X --dagnn_agg=$AGG \
      --dagnn_out_pool_all=$POOL_ALL --dagnn_out_pool=$POOL --dagnn_dropout=$DROPOUT \
      --checkpointing=1 --checkpoint=$CHECKPOINT  --patience=$PAT

echo "Run completed at:- "
date

