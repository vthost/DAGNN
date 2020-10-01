import sys
import os.path
from scipy.spatial.distance import pdist
from sparse_gp import SparseGP
import scipy.stats as sps
from scipy.io import loadmat
from scipy.stats import pearsonr
sys.path.append('%s/../software/enas' % os.path.dirname(os.path.realpath(__file__))) 
sys.path.append('%s/..' % os.path.dirname(os.path.realpath(__file__))) 
sys.path.insert(0, '../')
from util import *
from evaluate_BN import Eval_BN
from shutil import copy

from src.constants import *


'''Experiment settings'''
parser = argparse.ArgumentParser(description='Bayesian optimization experiments.')
# must specify
parser.add_argument('--data-name', default='final_structures6', help='graph dataset name')
parser.add_argument('--save-appendix', default='DVAE_PYG',
                    help='what is appended to data-name as save-name for results')
parser.add_argument('--checkpoint', type=int, default=10,
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
parser.add_argument('--BO-rounds', type=int, default=1,  #500,
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
parser.add_argument('--model', default='DVAE_PYG', help='model to use: DVAE, SVAE, \
                    DVAE_fast, DVAE_BN, SVAE_oneshot, DVAE_GCN')
parser.add_argument('--hs', type=int, default=501, metavar='N',
                    help='hidden size of GRUs')
parser.add_argument('--nz', type=int, default=56, metavar='N',
                    help='number of dimensions of latent vectors z')
parser.add_argument('--bidirectional', action='store_true', default=False,
                    help='whether to use bidirectional encoding')

parser.add_argument('--dagnn_layers', type=int, default=1)
parser.add_argument('--dagnn_agg_x', type=int, default=0, choices=[0, 1])
parser.add_argument('--dagnn_agg', type=str, default=NA_ATTN_X)
parser.add_argument('--dagnn_out_pool_all', type=int, default=0, choices=[0, 1])
parser.add_argument('--dagnn_out_pool', type=str, default=P_ADD, choices=[P_ATTN, P_CNN, P_MAX, P_MEAN, P_ADD])
parser.add_argument('--dagnn_dropout', type=float, default=0.0)

args = parser.parse_args()
data_name = args.data_name
save_appendix = args.save_appendix
data_dir = os.path.join(args.data_dir,'{}_{}'.format(data_name, save_appendix)) + "/" # data and model folder
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

data = loadmat(data_dir + '{}_latent_epoch{}.mat'.format(data_name, checkpoint))  # load train/test data

# do BO experiments with 10 random seeds
for rand_idx in range(1, 11):
    print("BO experiments - rand_idx:", rand_idx)

    save_dir = '{}results_{}_{}/'.format(res_dir, save_appendix, rand_idx)  # where to save the BO results
    if data_type == 'BN':
        eva = Eval_BN(save_dir)  # build the BN evaluator

    if not os.path.exists(save_dir):
        os.makedirs(save_dir) 

    # backup files
    copy('bo.py', save_dir)
    if args.predictor:
        copy('run_pred_{}.sh'.format(data_type), save_dir)
    elif args.vis_2d:
        copy('run_vis_{}.sh'.format(data_type), save_dir)
    else:
        copy('run_bo_{}.sh'.format(data_type), save_dir)

    # set seed
    random_seed = rand_idx
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed(random_seed)
    np.random.seed(random_seed)

    # load the data
    X_train = data['Z_train']
    y_train = -data['Y_train'].reshape((-1,1))
    if data_type == 'BN':
        # remove duplicates, otherwise SGP ill-conditioned
        #X_train, unique_idxs = np.unique(X_train, axis=0, return_index=True)
        #y_train = y_train[unique_idxs]
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

    while iteration < BO_rounds:
        print("Iteration", iteration)
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

        pred, uncert = sgp.predict(X_train, 0 * X_train)
        error = np.sqrt(np.mean((pred - y_train)**2))
        trainll = np.mean(sps.norm.logpdf(pred - y_train, scale = np.sqrt(uncert)))
        print('Train RMSE: ', error)
        print('Train ll: ', trainll)

        iteration += 1

