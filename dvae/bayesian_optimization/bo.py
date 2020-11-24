import pdb
import pickle
import sys
import os
import os.path
import collections
import torch
from tqdm import tqdm
import itertools
from scipy.spatial.distance import pdist
import matplotlib.pyplot as plt
from sparse_gp import SparseGP
import scipy.stats as sps
import numpy as np
import scipy.io
from scipy.io import loadmat
from scipy.stats import pearsonr
sys.path.append('%s/../software/enas' % os.path.dirname(os.path.realpath(__file__))) 
sys.path.append('%s/..' % os.path.dirname(os.path.realpath(__file__))) 
sys.path.insert(0, '../')

from util import *
from evaluate_BN import Eval_BN
from models import *
from dagnn import DAGNN
from dagnn_bn import DAGNN_BN
from src.constants import *
# m='DVAE_BN' #47.095159912109374
# m='DVAE_BN_PYG' #loss: 47.095159912109374
# m='DAGNN_BN'   # 46.940618896484374
# d='asia_200k'
#
# # m='DVAE_PYG' #
# # m='DVAE'
# m='DAGNN'
# d ='final_structures6'

'''Experiment settings'''
parser = argparse.ArgumentParser(description='Bayesian optimization experiments.')
# must specify
parser.add_argument('--data-name', default='asia_200k', help='graph dataset name')  #'final_structures6'
parser.add_argument('--save-appendix', default='DVAE_PYG',
                    help='what is appended to data-name as save-name for results')
parser.add_argument('--checkpoint', type=int, default=50,
                    help="load which epoch's model checkpoint")
parser.add_argument('--res-dir', default='../results-bo/',
                    help='where to save the Bayesian optimization results')
parser.add_argument('--data-dir', default='../results/',
                    help='where to save the Bayesian optimization results')
# BO settings
parser.add_argument('--predictor', action='store_true', default=False,
                    help='if True, use the performance predictor instead of SGP')
parser.add_argument('--grad-ascent', action='store_true', default=False,
                    help='if True and predictor=True, perform gradient-ascent with predictor')
parser.add_argument('--BO-rounds', type=int, default=10,
                    help="how many rounds of BO to perform")
parser.add_argument('--BO-batch-size', type=int, default=50, 
                    help="how many data points to select in each BO round")
parser.add_argument('--sample-dist', default='uniform', 
                    help='from which distrbiution to sample random points in the latent \
                    space as candidates to select; uniform or normal')
parser.add_argument('--random-baseline', action='store_true', default=False,
                    help='whether to include a baseline that randomly selects points \
                    to compare with Bayesian optimization')
parser.add_argument('--random-as-train', action='store_true', default=False,
                    help='if true, no longer use original train data to initialize SGP \
                    but randomly generates 1000 initial points as train data')
parser.add_argument('--random-as-test', action='store_true', default=False,
                    help='if true, randomly generates 100 points from the latent space \
                    as the additional testing data')
parser.add_argument('--vis-2d', action='store_true', default=False,
                    help='do visualization experiments on 2D space')


# can be inferred from the cmd_input.txt file, no need to specify
parser.add_argument('--data-type', default='BN',
                    help='ENAS: ENAS-format CNN structures; BN: Bayesian networks')
parser.add_argument('--model', default='DAGNN_BN', help='model to use: DVAE, SVAE, \
                    DVAE_fast, DAGNN_BN, SVAE_oneshot, DVAE_GCN')
parser.add_argument('--hs', type=int, default=501, metavar='N',
                    help='hidden size of GRUs')
parser.add_argument('--nz', type=int, default=56, metavar='N',
                    help='number of dimensions of latent vectors z')
parser.add_argument('--bidirectional', action='store_true', default=False,
                    help='whether to use bidirectional encoding')
parser.add_argument('--bo',type=int, default=0, choices=[0, 1],
                    help='whether to do BO')


parser.add_argument('--dagnn_layers', type=int, default=2)
parser.add_argument('--dagnn_agg', type=str, default=NA_ATTN_H)
parser.add_argument('--dagnn_out_wx', type=int, default=0, choices=[0, 1])
parser.add_argument('--dagnn_out_pool_all', type=int, default=0, choices=[0, 1])
parser.add_argument('--dagnn_out_pool', type=str, default=P_MAX, choices=[P_ATTN, P_MAX, P_MEAN, P_ADD])
parser.add_argument('--dagnn_dropout', type=float, default=0.0)

args = parser.parse_args()
data_name = args.data_name
save_appendix = args.save_appendix
data_dir = os.path.join(args.data_dir,'{}_{}'.format(data_name, save_appendix)) + "/"  # data and model folder
checkpoint = args.checkpoint
res_dir = args.res_dir
data_type = args.data_type
model_name = args.model
hs, nz = args.hs, args.nz
bidir = args.bidirectional
vis_2d = args.vis_2d

'''Load hyperparameters'''
with open(data_dir + 'cmd_input.txt', 'r') as f:
    cmd_input = f.readline()
cmd_input = cmd_input.split('--')
cmd_dict = {}
for z in cmd_input:
    z = z.split()
    if len(z) == 2:
        cmd_dict[z[0]] = z[1]
    elif len(z) == 1:
        cmd_dict[z[0]] = True
for key, val in cmd_dict.items():
    if key == 'data-type':
        data_type = val
    elif key == 'model':
        model_name = val
    elif key == 'hs':
        hs = int(val)
    elif key == 'nz':
        nz = int(val)

'''Load graph_args'''
with open(data_dir + data_name + '.pkl', 'rb') as f:
    _, _, graph_args = pickle.load(f)
START_TYPE, END_TYPE = graph_args.START_TYPE, graph_args.END_TYPE
max_n = graph_args.max_n
nvt = graph_args.num_vertex_type
args.nvt = nvt

'''BO settings'''
BO_rounds = args.BO_rounds
batch_size = args.BO_batch_size
sample_dist = args.sample_dist
random_baseline = args.random_baseline 
random_as_train = args.random_as_train
random_as_test = args.random_as_test

# other BO hyperparameters
lr = 0.0005  # the learning rate to train the SGP model
max_iter = 100  # how many iterations to optimize the SGP each time

# architecture performance evaluator
# if data_type == 'ENAS':
#     sys.path.append('%s/../software/enas/src/cifar10' % os.path.dirname(os.path.realpath(__file__)))
#     from evaluation import *
#     eva = Eval_NN()  # build the network acc evaluater
                     # defined in ../software/enas/src/cifar10/evaluation.py

data = loadmat(data_dir + '{}_latent_epoch{}.mat'.format(data_name, checkpoint))  # load train/test data
#data = loadmat(data_dir + '{}_latent.mat'.format(data_name))  # load train/test data

if torch.cuda.is_available():
    device = torch.device("cuda:" + str(0))
else:
    device = torch.device("cpu")

# do BO experiments with 10 random seeds
for rand_idx in range(1, 11):
    print("BO experiments - rand_idx:", rand_idx)

    save_dir = '{}results_{}_{}/'.format(res_dir, save_appendix, rand_idx)  # where to save the BO results
    if data_type == 'BN':
        eva = Eval_BN(save_dir)  # build the BN evaluator

    if not os.path.exists(save_dir):
        os.makedirs(save_dir) 

    # set seed
    random_seed = rand_idx
    torch.manual_seed(random_seed)
    np.random.seed(random_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(random_seed)

    # load the decoder
    if args.model.startswith("DAGNN"):
        model = eval(args.model)(nvt, args.hs, args.hs,
                                 graph_args.max_n,
                                 graph_args.num_vertex_type,
                                 graph_args.START_TYPE,
                                 graph_args.END_TYPE,
                                 hs=args.hs,
                                 nz=args.nz,
                                 num_nodes=nvt,
                                 agg=args.dagnn_agg,
                                 num_layers=args.dagnn_layers, bidirectional=args.bidirectional,
                                 out_wx=args.dagnn_out_wx > 0, out_pool_all=args.dagnn_out_pool_all,
                                 out_pool=args.dagnn_out_pool,
                                 dropout=args.dagnn_dropout)
    else:
        model = eval(args.model)(
            graph_args.max_n,
            graph_args.num_vertex_type,
            graph_args.START_TYPE,
            graph_args.END_TYPE,
            hs=args.hs,
            nz=args.nz,
            bidirectional=args.bidirectional
        )
    model.to(device)
    load_module_state(model, data_dir + 'model_checkpoint{}.pth'.format(checkpoint))

    # load the data
    X_train = data['Z_train']
    y_train = -data['Y_train'].reshape((-1,1))
    if data_type == 'BN':
        # remove duplicates, otherwise SGP ill-conditioned
        X_train, unique_idxs = np.unique(X_train, axis=0, return_index=True)
        y_train = y_train[unique_idxs]
        random_shuffle = np.random.permutation(range(len(X_train)))
        keep = 5000
        X_train = X_train[random_shuffle[:keep]]
        y_train = y_train[random_shuffle[:keep]]

    mean_y_train, std_y_train = np.mean(y_train), np.std(y_train)
    print('Mean, std of y_train is ', mean_y_train, std_y_train)
    y_train = (y_train - mean_y_train) / std_y_train
    X_test = data['Z_test']
    y_test = -data['Y_test'].reshape((-1,1))
    y_test = (y_test - mean_y_train) / std_y_train
    best_train_score = min(y_train)
    save_object((mean_y_train, std_y_train), "{}mean_std_y_train.dat".format(save_dir))

    print("Best train score is: ", best_train_score)

    '''Bayesian optimiation begins here'''
    iteration = 0
    best_score = 1e15
    best_arc = None
    best_random_score = 1e15
    best_random_arc = None
    print("Average pairwise distance between train points = {}".format(np.mean(pdist(X_train))))
    print("Average pairwise distance between test points = {}".format(np.mean(pdist(X_test))))

    if os.path.exists(save_dir + 'Test_RMSE_ll.txt'):
        os.remove(save_dir + 'Test_RMSE_ll.txt')
    if os.path.exists(save_dir + 'best_arc_scores.txt'):
        os.remove(save_dir + 'best_arc_scores.txt')

    while iteration < BO_rounds and args.bo:
        print("Iteration", iteration)
        if args.predictor:
            pred = model.predictor(torch.FloatTensor(X_test).to(device))
            pred = pred.detach().cpu().numpy()
            pred = (-pred - mean_y_train) / std_y_train
            uncert = np.zeros_like(pred)
        else:
            # We fit the GP
            M = 500
            sgp = SparseGP(X_train, 0 * X_train, y_train, M)
            sgp.train_via_ADAM(X_train, 0 * X_train, y_train, X_test, X_test * 0,  \
                y_test, minibatch_size = 2 * M, max_iterations = max_iter, learning_rate = lr)
            pred, uncert = sgp.predict(X_test, 0 * X_test)

        print("predictions: ", pred.reshape(-1))
        print("real values: ", y_test.reshape(-1))
        error = np.sqrt(np.mean((pred - y_test)**2))
        testll = np.mean(sps.norm.logpdf(pred - y_test, scale = np.sqrt(uncert)))
        print('Test RMSE: ', error)
        print('Test ll: ', testll)
        pearson = float(pearsonr(pred.flatten(), y_test.flatten())[0])
        print('Pearson r: ', pearson)
        with open(save_dir + 'Test_RMSE_ll.txt', 'a') as test_file:
            test_file.write('Test RMSE: {:.4f}, ll: {:.4f}, Pearson r: {:.4f}\n'.format(error, testll, pearson))

        error_if_predict_mean = np.sqrt(np.mean((np.mean(y_train, 0) - y_test)**2))
        print('Test RMSE if predict mean: ', error_if_predict_mean)
        if args.predictor:
            pred = model.predictor(torch.FloatTensor(X_train).to(device))
            pred = pred.detach().cpu().numpy()
            pred = (-pred - mean_y_train) / std_y_train
            uncert = np.zeros_like(pred)
        else:
            pred, uncert = sgp.predict(X_train, 0 * X_train)
        error = np.sqrt(np.mean((pred - y_train)**2))
        trainll = np.mean(sps.norm.logpdf(pred - y_train, scale = np.sqrt(uncert)))
        print('Train RMSE: ', error)
        print('Train ll: ', trainll)

        if args.bo:
            next_inputs = sgp.batched_greedy_ei(batch_size, np.min(X_train, 0), np.max(X_train, 0), np.mean(X_train, 0), np.std(X_train, 0), sample=sample_dist)
            valid_arcs_final = decode_from_latent_space(torch.FloatTensor(next_inputs).to(device), model,
                                                        500, max_n, False, data_type)

            new_features = next_inputs
            print("Evaluating selected points")
            scores = []
            for i in range(len(valid_arcs_final)):
                arc = valid_arcs_final[ i ]
                if arc is not None:
                    score = -eva.eval(arc)
                    score = (score - mean_y_train) / std_y_train
                else:
                    score = max(y_train)[ 0 ]
                if score < best_score:
                    best_score = score
                    best_arc = arc
                scores.append(score)
                # print(i, score)
            # print("Iteration {}'s selected arcs' scores:".format(iteration))
            # print(scores, np.mean(scores))
            save_object(scores, "{}scores{}.dat".format(save_dir, iteration))
            save_object(valid_arcs_final, "{}valid_arcs_final{}.dat".format(save_dir, iteration))

            if len(new_features) > 0:
                X_train = np.concatenate([ X_train, new_features ], 0)
                y_train = np.concatenate([ y_train, np.array(scores)[ :, None ] ], 0)
            #
            # print("Current iteration {}'s best score: {}".format(iteration, - best_score * std_y_train - mean_y_train))
            if best_arc is not None: # and iteration == 10:
                print("Best architecture: ", best_arc)
                with open(save_dir + 'best_arc_scores.txt', 'a') as score_file:
                    score_file.write(best_arc + ', {:.4f}\n'.format(-best_score * std_y_train - mean_y_train))
                if data_type == 'ENAS':
                    row = [int(x) for x in best_arc.split()]
                    g_best, _ = decode_ENAS_to_igraph(flat_ENAS_to_nested(row, max_n-2))
                elif data_type == 'BN':
                    row = adjstr_to_BN(best_arc)
                    g_best, _ = decode_BN_to_igraph(row)
                plot_DAG(g_best, save_dir, 'best_arc_iter_{}'.format(iteration), data_type=data_type, pdf=True)
        #
        iteration += 1
        # print(iteration)

# pdb.set_trace()
