# DAGNN
A graph neural network tailored to directed acyclic graphs that outperforms conventional GNNs by leveraging the partial order as strong inductive bias besides other suitable architectural features. <br/>Source code for ICLR 2021 paper https://openreview.net/forum?id=JbuYF437WB6.

**Update Dec'2023**<br/>Please also check out [our novel work](https://proceedings.neurips.cc/paper_files/paper/2023/file/94e85561a342de88b559b72c9b29f638-Paper-Conference.pdf), which proposes a more efficient, transformer-based model! The code can be found [here](https://github.com/LUOyk1999/DAGformer).

## Installation
* Tested with Python 3.7, PyTorch 1.5.0, and PyTorch Geometric 1.6.0
* Set up an Anaconda environment: `./setup.sh` 
<br/>(set the CUDA variables in the script)
* Alternatively, install the above and the packages listed in [requirements.txt](requirements.txt)

## Overview

* `/dvae` 
<br/>Basically the experiment code and data from [D-VAE](https://github.com/muhanzhang/D-VAE/).
The DAGNN models in this directory contain additional methods to support the decoding loss. 
* `/ogb` <br/>Basically the code from the [Open Graph Benchmark (OGB)](https://github.com/snap-stanford/ogb). We just added additional data preprocessing steps necessary for DAGNN.
* `/ogbg-code` <br/>Experiment code for TOK and LP experiments over the `ogbg-code2` data from OGB. The [DAGNN model](ogbg-code/model/dagnn.py) in this directory, can be considered as the basic implementation of DAGNN.
* `/scripts` <br/>Scripts for running the experiments.
* `/src` <br/> Basic utilities used in all experiments.

## OGB Experiments

* Run `./scripts/ogb_tok.sh` or `./scripts/ogb_lp.sh`
* The `ogbg-code2` data will be downloaded at first run, set the directory for storage in the scripts (`$DATA`). We updated the code since the ogbg-code dataset used in the paper is deprecated and not available anymore.
* By default, the script will run DAGNN over a 15% random subset of `ogbg-code2`. To change these settings, see the comments in the scripts.

## DAG Scoring Experiments
* The two datasets are Neural Architectures (NA) and Bayesian Networks (BN) in `/dvae/data`. Since the latter is rather large, we suggest to start with the former. 
* Run `./scripts/na_train.sh [DEVICE_ID] [MODEL]` for training and  `./scripts/na_eval.sh [DEVICE_ID] [MODEL] [EVALUATION_POINT]` for evaluation. And similar for BN.
<br/>`[MODEL]` can be one of: `DAGNN, DAGNN_BN, SVAE_GraphRNN, DVAE_GCN, DVAE_DeepGMG, DVAE, DVAE_BN`.
* For example `./scripts/na_train.sh 0 DAGNN`, `./scripts/na_eval.sh 0 DAGNN 100`.
* Note that the learning rates and epoch numbers in the scripts are the ones which we used for DAGNN. The experiment parameters for the D-VAE models and baselines can be found [here](https://github.com/muhanzhang/D-VAE/blob/master/commands.sh).

Please leave an issue if you have any trouble running the code or suggestions for improvements.
