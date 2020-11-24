from __future__ import print_function
import datetime
from torch import nn, optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
import scipy.io
from random import shuffle
from util import *
from models import *
from dagnn import DAGNN
from dagnn_bn import DAGNN_BN
from src.constants import *
import copy

# m='DAGNN_BN'
# m='DVAE_BN' #47.095159912109374
# m='DVAE_BN_PYG' #loss: 47.095159912109374
# m='DAGNN_BN'   # 46.940618896484374
# d='asia_200k'

# m='DVAE_PYG' #
# # m='DVAE'
m='DAGNN'
d ='final_structures6'

parser = argparse.ArgumentParser(description='Train Variational Autoencoders for DAGs')
# general settings
parser.add_argument('--data-type', default='BN' if d == 'asia_200k' else "ENAS",
                    help='ENAS: ENAS-format CNN structures; BN: Bayesian networks')
parser.add_argument('--data-name', default=d, help='graph dataset name')  # default='final_structures6',
parser.add_argument('--nvt', type=int, default=8 if d == 'asia_200k' else 6, help='number of different node types, \
                    6 for final_structures6, 8 for asia_200k')
parser.add_argument('--save-appendix', default='_'+m,
                    help='what to append to data-name as save-name for results')
parser.add_argument('--save-interval', type=int, default=10, metavar='N',  # 100
                    help='how many epochs to wait each time to save model states')
parser.add_argument('--sample-number', type=int, default=20, metavar='N',
                    help='how many samples to generate each time')
parser.add_argument('--no-test', action='store_true', default=False,
                    help='if True, merge test with train, i.e., no held-out set')
parser.add_argument('--reprocess', action='store_true', default=False,
                    help='if True, reprocess data instead of using prestored .pkl data')
parser.add_argument('--keep-old', action='store_true', default=True,
                    help='if True, do not remove any old data in the result folder')
parser.add_argument('--only-test', action='store_true', default=False,
                    help='if True, perform some experiments without training the model')
parser.add_argument('--small-train', action='store_true', default=False,
                    help='if True, use a smaller version of train set')
# model settings
parser.add_argument('--model', default=m, help='model to use: DVAE, SVAE, \
                    DVAE_fast, DVAE_BN, SVAE_oneshot, DVAE_GCN, DVAE_BN_PYG')
parser.add_argument('--load-latest-model', action='store_true', default=False,
                    help='whether to load latest_model.pth')
parser.add_argument('--continue-from', type=int, default=None, 
                    help="from which epoch's checkpoint to continue training")
parser.add_argument('--hs', type=int, default=501, metavar='N',
                    help='hidden size of GRUs')
parser.add_argument('--nz', type=int, default=56, metavar='N',
                    help='number of dimensions of latent vectors z')
parser.add_argument('--bidirectional', action='store_true', default=False if d == 'asia_200k' else True,  # USE ONLY WITH NNs as in paper
                    help='whether to use bidirectional encoding')
parser.add_argument('--predictor', action='store_true', default=False,
                    help='whether to train a performance predictor from latent\
                    encodings and a VAE at the same time')
# optimization settings
parser.add_argument('--lr', type=float, default=1e-4, metavar='LR',
                    help='learning rate (default: 1e-4)')
parser.add_argument('--epochs', type=int, default=10, metavar='N',  #0000
                    help='number of epochs to train')
parser.add_argument('--batch-size', type=int, default=2, metavar='N',
                    help='batch size during training')
parser.add_argument('--infer-batch-size', type=int, default=128, metavar='N',
                    help='batch size during inference')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='disables CUDA training')
parser.add_argument('--all-gpus', action='store_true', default=False,
                    help='use all available GPUs')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')

# parser.add_argument('--dagnn_agg_x', type=int, default=1 if d == 'asia_200k' else 0, choices=[0, 1])
# BN agg_x works not with max/sum, attn
# gated works, and only works with agg_x (because of parent layers used)
# attn x/h works with non-agg_x

parser.add_argument('--dagnn_layers', type=int, default=2)
parser.add_argument('--dagnn_agg', type=str, default=NA_ATTN_H)
parser.add_argument('--dagnn_out_wx', type=int, default=0, choices=[0, 1])
parser.add_argument('--dagnn_out_pool_all', type=int, default=0 if d == 'asia_200k' else 0, choices=[0, 1])
parser.add_argument('--dagnn_out_pool', type=str, default=P_MAX, choices=[P_ATTN, P_MAX, P_MEAN, P_ADD])
parser.add_argument('--dagnn_dropout', type=float, default=0.0)

parser.add_argument('--clip', default=0, type=float,
                    help='...')
parser.add_argument('--device', type=int, default=0,
                    help='')
parser.add_argument('--res_dir', type=str, default="",
                    help='')

args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()
torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)
    device = torch.device("cuda:"+str(args.device))
else:
    device = torch.device("cpu")
np.random.seed(args.seed)
random.seed(args.seed)
print(args)


'''Prepare data'''
args.file_dir = os.path.dirname(os.path.realpath('__file__'))
args.res_dir = args.res_dir if args.res_dir else os.path.join(args.file_dir, 'results')
args.res_dir = os.path.join(args.res_dir, '{}{}'.format(args.data_name, args.save_appendix))
if not os.path.exists(args.res_dir):
    os.makedirs(args.res_dir) 

pkl_name = os.path.join(args.res_dir, args.data_name + '.pkl')

# check whether to load pre-stored pickle data
if os.path.isfile(pkl_name) and not args.reprocess:
    with open(pkl_name, 'rb') as f:
        train_data, test_data, graph_args = pickle.load(f)
# otherwise process the raw data and save to .pkl
else:
    # determine data formats according to models, DVAE: igraph, SVAE: string (as tensors)
    # DAGNN
    if "PYG" in args.model or "DAGNN" in args.model:
        input_fmt = 'pyg'
    elif args.model.startswith('DVAE'):
        input_fmt = 'igraph'
    elif args.model.startswith('SVAE'):
        input_fmt = 'string'
    if args.data_type == 'ENAS':
        train_data, test_data, graph_args = load_ENAS_graphs(args.data_name, n_types=args.nvt,
                                                             fmt=input_fmt)
    elif args.data_type == 'BN':
        train_data, test_data, graph_args = load_BN_graphs(args.data_name, n_types=args.nvt,
                                                           fmt=input_fmt)
    with open(pkl_name, 'wb') as f:
        pickle.dump((train_data, test_data, graph_args), f)


# save command line input
cmd_input = 'python ' + ' '.join(sys.argv) + '\n'
with open(os.path.join(args.res_dir, 'cmd_input.txt'), 'a') as f:
    f.write(cmd_input)
print('Command line input: ' + cmd_input + ' is saved.')

# construct train data
if args.no_test:
    train_data = train_data + test_data

if args.small_train:
    train_data = train_data[:100]


'''Prepare the model'''
# model
if args.model.startswith("DAGNN"):
    model = eval(args.model)(args.nvt + 2, args.hs, args.hs,
                  graph_args.max_n,
                  graph_args.num_vertex_type,
                  graph_args.START_TYPE,
                  graph_args.END_TYPE,
                  hs=args.hs,
                  nz=args.nz,
                  num_nodes=args.nvt+2,
                  agg=args.dagnn_agg,
                  num_layers=args.dagnn_layers, bidirectional=args.bidirectional,
                  out_wx=args.dagnn_out_wx > 0, out_pool_all=args.dagnn_out_pool_all, out_pool=args.dagnn_out_pool,
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
if args.predictor:
    predictor = nn.Sequential(
            nn.Linear(args.nz, args.hs), 
            nn.Tanh(), 
            nn.Linear(args.hs, 1)
            )
    model.predictor = predictor
    model.mseloss = nn.MSELoss(reduction='sum')
# optimizer and scheduler
optimizer = optim.Adam(model.parameters(), lr=args.lr)
scheduler = ReduceLROnPlateau(optimizer, 'min', factor=0.1, patience=10, verbose=True)

model.to(device)

if args.all_gpus:
    net = custom_DataParallel(model, device_ids=range(torch.cuda.device_count()))

if args.load_latest_model:
    ts = []
    for fn in os.listdir(args.res_dir):
        if "model_checkpoint" in fn:
            ts += [int(fn[16:fn.rindex(".")])]
    if ts:
        args.continue_from = max(ts)

if args.continue_from is not None:
    epoch = args.continue_from
    load_module_state(model, os.path.join(args.res_dir, 'model_checkpoint{}.pth'.format(epoch)))
    load_module_state(optimizer, os.path.join(args.res_dir, 'optimizer_checkpoint{}.pth'.format(epoch)))
    load_module_state(scheduler, os.path.join(args.res_dir, 'scheduler_checkpoint{}.pth'.format(epoch)))
    print("Loaded module_state epoch", epoch)


'''Define some train/test functions'''
def train(epoch, args):
    model.train()
    train_loss = 0
    recon_loss = 0
    kld_loss = 0
    pred_loss = 0
    shuffle(train_data)
    pbar = tqdm(train_data)
    g_batch = []
    y_batch = []
    for i, (g, y) in enumerate(pbar):
        if args.model.startswith('SVAE'):  # for SVAE, g is tensor
            g = g.to(device)
        g_batch.append(g)
        y_batch.append(y)
        if len(g_batch) == args.batch_size or i == len(train_data) - 1:
            optimizer.zero_grad()
            g_batch = model._collate_fn(g_batch)
            if args.all_gpus:  # does not support predictor yet
                loss = net(g_batch).sum()
                pbar.set_description('Epoch: %d, loss: %0.4f' % (epoch, loss.item()/len(g_batch)))
                recon, kld = 0, 0
            else:
                mu, logvar = model.encode(g_batch)
                loss, recon, kld = model.loss(mu, logvar, g_batch)
                if args.predictor:
                    y_batch = torch.FloatTensor(y_batch).unsqueeze(1).to(device)
                    y_pred = model.predictor(mu)
                    pred = model.mseloss(y_pred, y_batch)
                    loss += pred
                    pbar.set_description('Epoch: %d, loss: %0.4f, recon: %0.4f, kld: %0.4f, pred: %0.4f'\
                            % (epoch, loss.item()/len(g_batch), recon.item()/len(g_batch), 
                            kld.item()/len(g_batch), pred/len(g_batch)))
                else:
                    pbar.set_description('Epoch: %d, loss: %0.4f, recon: %0.4f, kld: %0.4f' % (
                                     epoch, loss.item()/len(g_batch), recon.item()/len(g_batch), 
                                     kld.item()/len(g_batch)))
            loss.backward()
            if args.clip > 0:
                torch.nn.utils.clip_grad_norm(model.parameters(), args.clip)
            
            train_loss += float(loss)
            recon_loss += float(recon)
            kld_loss += float(kld)
            if args.predictor:
                pred_loss += float(pred)
            optimizer.step()
            g_batch = []
            y_batch = []

    print('====> Epoch: {} Average loss: {:.4f}'.format(
          epoch, train_loss / len(train_data)))

    if args.predictor:
        return train_loss, recon_loss, kld_loss, pred_loss
    return train_loss, recon_loss, kld_loss


def test():
    # test recon accuracy
    model.eval()
    encode_times = 10
    decode_times = 10
    Nll = 0
    pred_loss = 0
    n_perfect = 0
    print('Testing begins...')
    pbar = tqdm(test_data)
    g_batch = []
    y_batch = []
    for i, (g, y) in enumerate(pbar):
        if args.model.startswith('SVAE'):
            g = g.to(device)
        g_batch.append(g)
        y_batch.append(y)
        if len(g_batch) == args.infer_batch_size or i == len(test_data) - 1:
            g = model._collate_fn(g_batch)
            mu, logvar = model.encode(g)
            _, nll, _ = model.loss(mu, logvar, g)
            pbar.set_description('nll: {:.4f}'.format(nll.item()/len(g_batch)))
            Nll += nll.item()
            g_batch = []
            y_batch = []
    Nll /= len(test_data)
    pred_loss /= len(test_data)
    pred_rmse = math.sqrt(pred_loss)
    acc = n_perfect / (len(test_data) * encode_times * decode_times)
    if args.predictor:
        print('Test average recon loss: {0}, recon accuracy: {1:.4f}, pred rmse: {2:.4f}'.format(
            Nll, acc, pred_rmse))
        return Nll, acc, pred_rmse
    else:
        print('Test average recon loss: {0}, recon accuracy: {1:.4f}'.format(Nll, acc))
        return Nll, acc


def extract_latent(data):
    model.eval()
    Z = []
    Y = []
    g_batch = []
    for i, (g, y) in enumerate(tqdm(data)):
        if args.model.startswith('SVAE'):
            g_ = g.to(device)
        elif args.model.startswith('DVAE') or args.model.startswith('DAGNN'):
            # copy graph
            # otherwise original igraphs will save the H states and consume more GPU memory
            g_ = copy.deepcopy(g)
        g_batch.append(g_)
        if len(g_batch) == args.infer_batch_size or i == len(data) - 1:
            g_batch = model._collate_fn(g_batch)
            mu, _ = model.encode(g_batch)
            mu = mu.cpu().detach().numpy()
            Z.append(mu)
            g_batch = []
        Y.append(y)
    return np.concatenate(Z, 0), np.array(Y)


'''Extract latent representations Z'''
def save_latent_representations(epoch):
    Z_train, Y_train = extract_latent(train_data)
    Z_test, Y_test = extract_latent(test_data)
    latent_pkl_name = os.path.join(args.res_dir, args.data_name +
                                   '_latent_epoch{}.pkl'.format(epoch))
    latent_mat_name = os.path.join(args.res_dir, args.data_name + 
                                   '_latent_epoch{}.mat'.format(epoch))
    with open(latent_pkl_name, 'wb') as f:
        pickle.dump((Z_train, Y_train, Z_test, Y_test), f)
    print('Saved latent representations to ' + latent_pkl_name)
    scipy.io.savemat(latent_mat_name, 
                     mdict={
                         'Z_train': Z_train, 
                         'Z_test': Z_test, 
                         'Y_train': Y_train, 
                         'Y_test': Y_test
                         }
                     )


time = datetime.datetime.now()

'''Training begins here'''
min_loss = math.inf  # >= python 3.5
min_loss_epoch = None
loss_name = os.path.join(args.res_dir, 'train_loss.txt')
loss_plot_name = os.path.join(args.res_dir, 'train_loss_plot.pdf')
test_results_name = os.path.join(args.res_dir, 'test_results.txt')
if os.path.exists(loss_name) and not args.keep_old:
    os.remove(loss_name)

if args.only_test:
    epoch = args.continue_from

start_epoch = args.continue_from if args.continue_from is not None else 0
for epoch in range(start_epoch + 1, args.epochs + 1):
    if args.predictor:
        train_loss, recon_loss, kld_loss, pred_loss = train(epoch, args)
    else:
        train_loss, recon_loss, kld_loss = train(epoch, args)
        pred_loss = 0.0
    with open(loss_name, 'a') as loss_file:
        loss_file.write("{} {:.4f} {:.4f} {:.4f}\n".format(
            epoch,
            train_loss/len(train_data), 
            recon_loss/len(train_data), 
            kld_loss/len(train_data), 
            # pred_loss/len(train_data),
            ))


    scheduler.step(train_loss)
    if epoch > 5 and ((epoch%args.save_interval) == 0) or epoch ==1:
        print("save current model...", epoch, args.save_interval, epoch%args.save_interval)
        model_name = os.path.join(args.res_dir, 'model_checkpoint{}.pth'.format(epoch))
        optimizer_name = os.path.join(args.res_dir, 'optimizer_checkpoint{}.pth'.format(epoch))
        scheduler_name = os.path.join(args.res_dir, 'scheduler_checkpoint{}.pth'.format(epoch))
        torch.save(model.state_dict(), model_name)
        torch.save(optimizer.state_dict(), optimizer_name)
        torch.save(scheduler.state_dict(), scheduler_name)
        print("visualize reconstruction examples...")
        # visualize_recon(epoch)
        print("extract latent representations...")
        save_latent_representations(epoch)


print("TRAIN TIME", datetime.datetime.now()-time)
'''Testing begins here'''
if args.predictor:
    Nll, acc, pred_rmse = test()
else:
    Nll, acc = test()
    pred_rmse = 0
r_valid, r_unique, r_novel = 0,0,0 #prior_validity(True)
with open(test_results_name, 'a') as result_file:
    result_file.write("Epoch {} Test recon loss: {} recon acc: {:.4f} r_valid: {:.4f}".format(
            epoch, Nll, acc, r_valid) + 
            " r_unique: {:.4f} r_novel: {:.4f} pred_rmse: {:.4f}\n".format(
            r_unique, r_novel, pred_rmse))

