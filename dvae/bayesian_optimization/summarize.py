import sys
import os.path
from shutil import copy
sys.path.append('%s/..' % os.path.dirname(os.path.realpath(__file__))) 
sys.path.insert(0, '../')
from util import *

'''
This script is for summarizing the Bayesian optimization and latent space
predictivity results.
'''


parser = argparse.ArgumentParser(description='...')
parser.add_argument('--name', type=str, default="",
                    help='exp id')
parser.add_argument('--res-dir', type=str, default="",
                    help='exp id')
parser.add_argument('--data-type', default='BN',
                    help='ENAS: ENAS-format CNN structures; BN: Bayesian networks')
args = parser.parse_args()


# Change experiment settings here

data_type = args.data_type
save_appendix = [args.name]
model_name = save_appendix
# save_appendix = ['DVAE', 'SVAE', 'GraphRNN', 'GCN', 'DeepGMG']
# model_name = ['D-VAE', 'S-VAE', 'GraphRNN', 'GCN', 'DeepGMG']

if data_type == 'ENAS':
    max_n = 8  # number of nodes
    res_dir = 'NN_results/' if not args.res_dir else args.res_dir + "/"
    n_iter = 10
    num_random_seeds = 10
    
    random_baseline = False  # whether to compare BO with random 
    include_train_scores = False  # whether to also include train scores when showing best selected arcs's scores
    random_as_test = False  # whether to use results on random test

elif data_type == 'BN':
    max_n = 10  # number of nodes
    res_dir = 'BN_results/' if not args.res_dir else args.res_dir+ "/"
    n_iter = 10
    num_random_seeds = 10

    random_baseline = False  # whether to compare BO with random 
    include_train_scores = False
    random_as_test = False


aggregate_dir = '{}_aggregate_results/'.format(data_type)

if not os.path.exists(aggregate_dir):
    os.makedirs(aggregate_dir) 
copy(os.path.realpath(__file__), aggregate_dir)

if random_as_test:
    test_res_file = 'Random_Test_RMSE_ll.txt'
else:
    test_res_file = 'Test_RMSE_ll.txt'

all_test_rmse, all_test_r = [[] for _ in model_name], [[] for _ in model_name]
for random_seed in range(1, num_random_seeds+1):
    save_dir = ['{}results_{}_{}/'.format(res_dir, x, random_seed) for x in save_appendix]  # where to load the BO results of first model
    if random_baseline:
        random_dir = ['{}results_{}_{}/'.format(res_dir, x, random_seed) for x in save_appendix]

    mean_y_train, std_y_train = [0] * len(model_name), [0] * len(model_name)
    for i, x in enumerate(save_appendix):
        mean_y_train[i], std_y_train[i] = load_object('{}results_{}_{}/mean_std_y_train.dat'.format(res_dir, x, random_seed))


    test_rmse, test_r = [[] for _ in model_name], [[] for _ in model_name]
    for i in range(len(model_name)):
        with open(save_dir[i] + test_res_file, 'r') as f:
            lines = f.readlines()
            for j, line in enumerate(lines):
                if j >= n_iter:
                    break
                blocks = line.split(',')
                rmse, ll, r = blocks[0][-6:], blocks[1][-6:], blocks[2][-6:]
                test_rmse[i].append(float(rmse))
                test_r[i].append(float(r))
        all_test_rmse[i].append(test_rmse[i])
        all_test_r[i].append(test_r[i])


for i in range(len(model_name)):
    all_test_rmse[i] = np.array(all_test_rmse[i])
    all_test_r[i] = np.array(all_test_r[i])

# plot best scores over time
def get_highest_over_time(scores):
    highest_mean = scores.max(2).mean(0)
    highest_std = scores.max(2).std(0)
    highest_so_far = [highest_mean[0]]
    std_so_far = [highest_std[0]]
    for i, x in enumerate(highest_mean):
        if i == 0:
            continue
        if x > highest_so_far[-1]:
            cm, cs = x, highest_std[i]
        else:
            cm, cs = highest_so_far[-1], std_so_far[-1]
        highest_so_far.append(cm)
        std_so_far.append(cs)
    return (highest_so_far, std_so_far)

f = open(aggregate_dir + 'output.txt', 'a')
for name, label in zip(['rmse', 'r'], ['RMSE', 'Pearson\'s r']):
    for i in range(len(model_name)):
        string = 'Model {0}, {1}, {2}, first iter, '.format(i, model_name[i], res_dir), name, eval('all_test_{1}[{0}]'.format(i, name)).mean(0)[0], eval('all_test_{}[{}]'.format(name, i)).std(0)[0]
        print(*string)
        print(*string, file=f)


f.close()

