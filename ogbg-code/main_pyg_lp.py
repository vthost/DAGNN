
import torch.optim as optim
import torch.nn.functional as F
from torchvision import transforms
from models.gnn import GNN
from tqdm import tqdm
import argparse
import numpy as np
import torch.multiprocessing
torch.multiprocessing.set_sharing_strategy('file_system')
### importing OGB
from ogb.graphproppred import Evaluator, PygGraphPropPredDataset

### for data transform
from utils import augment_edge
### DAGNN
import random
from models.dagnn import DAGNN
from models.gnn2 import GAT, GGNN, SAGPoolGNN
from models.asap import ASAP
from src.constants import *
from torch_geometric.data import DataLoader
from src.tg.dataloader import DataLoader
from src.tg.data_parallel import DataParallel
###
import pandas as pd
# make sure summary_report is imported after src.utils (also from dependencies)
from utils2 import *


multicls_criterion = torch.nn.CrossEntropyLoss()


def train(model, device, loader, optimizer, args, evaluator):
    model.train()

    y_true = []
    y_pred = []
    loss_accum = 0
    for step, batch in enumerate(tqdm(loader, desc="Iteration")):
        # one b(atch) per device
        batch = [b for b in batch if not b.x.shape[0] == 1 and not b.batch[-1] == 0 ]
        if batch:
            pred = model(batch)
            optimizer.zero_grad()

            targ = torch.cat([b.len_longest_path.to(device) for b in batch], dim=0)

            loss = multicls_criterion(pred, targ.to(torch.long))  #batch.y.view(-1, ))
            loss.backward()
            if args.clip > 0:
                torch.nn.utils.clip_grad_norm(model.parameters(), args.clip)
            optimizer.step()

            loss_accum += loss.item()

            y_true.append(targ.view(-1,1).detach().cpu())
            y_pred.append(torch.argmax(pred.detach(), dim=1).view(-1, 1).cpu())

    y_true = torch.cat(y_true, dim=0).numpy()
    y_pred = torch.cat(y_pred, dim=0).numpy()
    # print(y_true)
    # print(y_pred)
    input_dict = {"y_true": y_true, "y_pred": y_pred}
    return loss_accum / (step + 1), evaluator.eval(input_dict)


def eval(model, device, loader, evaluator):
    model.eval()

    y_true = []
    y_pred = []
    for step, batch in enumerate(tqdm(loader, desc="Iteration")):
        # one b(atch) per device
        batch = [b for b in batch if not b.x.shape[0] == 1]
        if batch:
            with torch.no_grad():
                pred = model(batch)

            y_true1 = [b.len_longest_path.view(-1,1).detach().cpu() for b in batch]
            y_true += y_true1

            y_pred.append(torch.argmax(pred.detach(), dim = 1).view(-1,1).cpu())


    y_pred = torch.cat(y_pred, dim=0)
    y_true = torch.cat(y_true, dim=0).view(y_pred.shape)
    y_true = y_true.numpy()
    y_pred = y_pred.numpy()
    # print(y_true)
    # print(y_pred)
    input_dict = {"y_true": y_true, "y_pred": y_pred}

    return evaluator.eval(input_dict)


def main():
    # Training settings
    parser = argparse.ArgumentParser(description='GNN baselines on ogbg-code data with Pytorch Geometrics')
    parser.add_argument('--device', type=int, default=0,
                        help='which gpu to use if any (default: 0)')
    parser.add_argument('--gnn', type=str, default="dagnn",
                        help='GNN gin, gin-virtual, or gcn, or gcn-virtual (default: gcn-virtual)')
    parser.add_argument('--drop_ratio', type=float, default=0,
                        help='dropout ratio (default: 0)')
    parser.add_argument('--max_seq_len', type=int, default=5,
                        help='maximum sequence length to predict (default: 5)')
    parser.add_argument('--num_vocab', type=int, default=5000,
                        help='the number of vocabulary used for sequence prediction (default: 5000)')
    parser.add_argument('--num_layer', type=int, default=5,
                        help='number of GNN message passing layers (default: 5)')
    parser.add_argument('--emb_dim', type=int, default=300,
                        help='dimensionality of hidden units in GNNs (default: 300)')
    parser.add_argument('--batch_size', type=int, default=20,
                        help='input batch size for training (default: 128)')
    parser.add_argument('--epochs', type=int, default=30,
                        help='number of epochs to train (default: 30)')
    parser.add_argument('--num_workers', type=int, default=0,
                        help='number of workers (default: 0)')
    parser.add_argument('--dataset', type=str, default="ogbg-code",
                        help='dataset name (default: ogbg-code)')

    parser.add_argument('--filename', type=str, default="test",
                        help='filename to output result (default: )')

    ### DAGNN
    parser.add_argument('--dagnn_wea', type=int, default=0, choices=[0, 1])
    parser.add_argument('--dagnn_layers', type=int, default=1)
    parser.add_argument('--dagnn_bidir', type=int, default=0, choices=[0, 1])
    parser.add_argument('--dagnn_agg_x', type=int, default=0, choices=[0, 1])
    parser.add_argument('--dagnn_agg', type=str, default=NA_GATED_SUM)
    parser.add_argument('--dagnn_out_pool_all', type=int, default=0, choices=[0, 1])
    parser.add_argument('--dagnn_out_pool', type=str, default=P_MAX, choices=[P_ATTN, P_CNN, P_MAX, P_MEAN, P_ADD])
    parser.add_argument('--dagnn_dropout', type=float, default=0.0)
    parser.add_argument('--dagnn_mapper_bias', type=int, default=1, choices=[0, 1])
    parser.add_argument('--dagnn_dense', type=int, default=0, choices=[0, 1])

    parser.add_argument('--dir_data', type=str, default=None,
                        help='... dir')
    parser.add_argument('--dir_results', type=str, default=DIR_LPRESULTS,
                        help='results dir')
    parser.add_argument('--dir_save', default=DIR_SAVED_MODELS,
                        help='directory to save checkpoints in')
    parser.add_argument('--train_idx', default="",
                        help='...')
    parser.add_argument('--checkpointing', default=1, type=int, choices=[0, 1],
                        help='...')
    parser.add_argument('--checkpoint', default="",
                        help='...')
    parser.add_argument('--folds', default=10, type=int,
                        help='...')
    parser.add_argument('--clip', default=0, type=float,
                        help='...')
    parser.add_argument('--lr', default=1e-3, type=float,
                        help='learning rate (default: 1e-3)')
    parser.add_argument('--patience', default=20, type=float,
                        help='learning rate (default: 1e-3)')
    ###

    args = parser.parse_args()
    device = torch.device("cuda:" + str(args.device)) if torch.cuda.is_available() else torch.device("cpu")

    os.makedirs(args.dir_results, exist_ok=True)
    os.makedirs(args.dir_save, exist_ok=True)

    train_file = os.path.join(args.dir_results, args.filename + '_train.csv')
    if not os.path.exists(train_file):
        with open(train_file, 'w') as f:
            f.write("fold,epoch,loss,train,valid,test\n")
    res_file = os.path.join(args.dir_results, args.filename + '.csv')
    if not os.path.exists(res_file):
        with open(res_file, 'w') as f:
            f.write("fold,epoch,bestv_train,bestv_valid,bestv_test\n")

    ### automatic dataloading and splitting
    dataset = PygGraphPropPredDataset(name=args.dataset, root="dataset" if args.dir_data is None else args.dir_data)
    dataset.eval_metric = "acc"
    dataset.task_type = "classification"

    split_idx = dataset.get_idx_split()

    if args.train_idx:
        train_idx = pd.read_csv(os.path.join("dataset", args.train_idx + ".csv.gz"), compression="gzip", header=None).values.T[0]
        train_idx = torch.tensor(train_idx, dtype = torch.long)
        split_idx['train'] = train_idx

    ### set the transform function
    # DAGNN
    augment = augment_edge2 if "dagnn" in args.gnn else augment_edge
    dataset.transform = transforms.Compose([augment])

    ### automatic acc evaluator. takes dataset name as input
    evaluator = Evaluator("ogbg-ppa")

    nodetypes_mapping = pd.read_csv(os.path.join(dataset.root, 'mapping', 'typeidx2type.csv.gz'))
    nodeattributes_mapping = pd.read_csv(os.path.join(dataset.root, 'mapping', 'attridx2attr.csv.gz'))
    ### Encoding node features into emb_dim vectors.
    ### The following three node features are used.
    # 1. node type
    # 2. node attribute
    # 3. node depth
    node_encoder = ASTNodeEncoder2(args.emb_dim, num_nodetypes = len(nodetypes_mapping['type']), num_nodeattributes = len(nodeattributes_mapping['attr']), max_depth = 20)

    start_fold = 1
    checkpoint_fn = ""
    train_results, valid_results, test_results = [], [], []     # on fold level

    if args.checkpointing and args.checkpoint:
        s = args.checkpoint[:-3].split("_")
        start_fold = int(s[-2])
        start_epoch = int(s[-1]) + 1

        checkpoint_fn = os.path.join(args.dir_save, args.checkpoint)  # need to remove it in any case

        if start_epoch > args.epochs:  # DISCARD checkpoint's model (ie not results), need a new model!
            args.checkpoint = ""
            start_fold += 1

            results = load_checkpoint_results(checkpoint_fn)
            train_results, valid_results, test_results, train_curve, valid_curve, test_curve = results

    # start
    for fold in range(start_fold, args.folds + 1):
        # fold-specific settings & data splits
        torch.manual_seed(fold)
        random.seed(fold)
        np.random.seed(fold)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(fold)
            torch.backends.cudnn.benchmark = True
            # torch.backends.cudnn.deterministic = True
            # torch.backends.cudnn.benchmark = False

        n_devices = torch.cuda.device_count() if torch.cuda.device_count() > 0 else 1
        train_loader = DataLoader(dataset[split_idx["train"]], batch_size=args.batch_size, shuffle=True,
                                  num_workers = args.num_workers, n_devices=n_devices)
        valid_loader = DataLoader(dataset[split_idx["valid"]], batch_size=args.batch_size, shuffle=False,
                                  num_workers=args.num_workers, n_devices=n_devices)
        test_loader = DataLoader(dataset[split_idx["test"]], batch_size=args.batch_size, shuffle=False,
                                 num_workers=args.num_workers, n_devices=n_devices)

        start_epoch = 1

        # model etc.
        model = init_model(args, node_encoder)

        print("Let's use", torch.cuda.device_count(), "GPUs! -- DataParallel running also on CPU only")
        device_ids = list(range(torch.cuda.device_count())) if torch.cuda.device_count() > 0 else None
        model = DataParallel(model, device_ids)
        model.to(device)

        optimizer = optim.Adam(model.parameters(), lr=args.lr)

        # overwrite some settings
        if args.checkpointing and args.checkpoint:
            # signal that it has been used
            args.checkpoint = ""

            results, start_epoch, model, optimizer = load_checkpoint(checkpoint_fn, model, optimizer)
            train_results, valid_results, test_results, train_curve, valid_curve, test_curve = results
            start_epoch += 1
        else:
            valid_curve, test_curve, train_curve = [], [], []

        # start new epoch
        for epoch in range(start_epoch, args.epochs + 1):
            old_checkpoint_fn = checkpoint_fn
            checkpoint_fn = '%s.pt' % os.path.join(args.dir_save, args.filename + "_" + str(fold) + "_" + str(epoch))

            print("=====Fold {}, Epoch {}".format(fold, epoch))
            loss, train_perf = train(model, device, train_loader, optimizer, args, evaluator)
            valid_perf = eval(model, device, valid_loader, evaluator)
            test_perf = eval(model, device, test_loader, evaluator)

            print({'Train': train_perf, 'Validation': valid_perf, 'Test': test_perf})
            with open(train_file, 'a') as f:
                f.write("{},{},{:.4f},{:.4f},{:.4f},{:.4f}\n".format(fold, epoch, loss, train_perf[dataset.eval_metric], valid_perf[dataset.eval_metric], test_perf[dataset.eval_metric]))

            train_curve.append(train_perf[dataset.eval_metric])
            valid_curve.append(valid_perf[dataset.eval_metric])
            test_curve.append(test_perf[dataset.eval_metric])

            ### DAGNN
            if args.checkpointing:
                create_checkpoint(checkpoint_fn, epoch, model, optimizer, (train_results, valid_results, test_results, train_curve, valid_curve, test_curve))
                if fold > 1 or epoch > 1:
                    remove_checkpoint(old_checkpoint_fn)

            best_val_epoch = np.argmax(np.array(valid_curve))
            if args.patience > 0 and best_val_epoch + 1 + args.patience < epoch:
                print("Early stopping!")
                break

        print('Finished training for fold {} !'.format(fold)+"*"*20)
        print('Best validation score: {}'.format(valid_curve[best_val_epoch]))
        print('Test score: {}'.format(test_curve[best_val_epoch]))

        with open(res_file, 'a') as f:
            results = [fold, best_val_epoch, train_curve[best_val_epoch], valid_curve[best_val_epoch],test_curve[best_val_epoch]]
            f.writelines(",".join([str(v) for v in results]) + "\n")

        train_results += [train_curve[best_val_epoch]]
        valid_results += [valid_curve[best_val_epoch]]
        test_results += [test_curve[best_val_epoch]]

        results = list(summary_report(train_results)) + list(summary_report(valid_results)) + list(summary_report(test_results))
        # with open(res_file, 'a') as f:
        #     f.writelines(str(fold)+ ",_," + ",".join([str(v) for v in results]) + "\n")
        print(",".join([str(v) for v in results]))

    results = list(summary_report(train_results)) + list(summary_report(valid_results)) + list(summary_report(test_results))
    with open(res_file, 'a') as f:
        f.writelines(str(fold) + ",_," + ",".join([str(v) for v in results]) + "\n")
        # print(",".join([str(v) for v in results]))

    # we might want to add folds
    # if args.checkpointing:
    #     remove_checkpoint(checkpoint_fn)


def init_model(args, node_encoder, numclass=275):
    # this was only relevant for regression version
    n = 1 #len(vocab2idx)
    m = 1 #args.max_seq_len
    if args.gnn == 'gin':
        model = GNN(num_vocab=n, max_seq_len=m, node_encoder=node_encoder,
                    num_layer=args.num_layer, gnn_type='gin', emb_dim=args.emb_dim, drop_ratio=args.drop_ratio,
                    virtual_node=False, num_class=numclass)
    elif args.gnn == 'gin-virtual':
        model = GNN(num_vocab=n, max_seq_len=m, node_encoder=node_encoder,
                    num_layer=args.num_layer, gnn_type='gin', emb_dim=args.emb_dim, drop_ratio=args.drop_ratio,
                    virtual_node=True, num_class=numclass)
    elif args.gnn == 'gcn':
        model = GNN(num_vocab=n, max_seq_len=m, node_encoder=node_encoder,
                    num_layer=args.num_layer, gnn_type='gcn', emb_dim=args.emb_dim, drop_ratio=args.drop_ratio,
                    virtual_node=False, num_class=numclass)
    elif args.gnn == 'gcn-virtual':
        model = GNN(num_vocab=n, max_seq_len=m, node_encoder=node_encoder,
                    num_layer=args.num_layer, gnn_type='gcn', emb_dim=args.emb_dim, drop_ratio=args.drop_ratio,
                    virtual_node=True, num_class=numclass)
    elif args.gnn == 'ggnn':
        model = GGNN(num_vocab=n, max_seq_len=m, node_encoder=node_encoder,
                     emb_dim=args.emb_dim, num_class=numclass)
    elif args.gnn == 'gat':
        model = GAT(num_vocab=n, max_seq_len=m, node_encoder=node_encoder,
                    emb_dim=args.emb_dim, num_layers=args.num_layer, num_class=numclass)
    elif args.gnn == 'sagpool':
        model = SAGPoolGNN(num_vocab=n, max_seq_len=m, node_encoder=node_encoder, emb_dim=args.emb_dim, num_layers=args.num_layer, num_class=numclass)
    elif args.gnn == 'asap':
        model = ASAP(n, m, node_encoder, args.emb_dim, args.num_layer,
                     args.emb_dim, num_class=numclass)

    elif "dagnn" in args.gnn:

        model = DAGNN(num_vocab=n, max_seq_len=m, emb_dim=args.emb_dim,
                   hidden_dim=args.emb_dim, out_dim=None, encoder=node_encoder,
                   w_edge_attr=args.dagnn_wea, num_layers=args.dagnn_layers, bidirectional=args.dagnn_bidir,
                   agg=args.dagnn_agg, agg_x=args.dagnn_agg_x > 0, mapper_bias=args.dagnn_mapper_bias > 0,
                   out_pool_all=args.dagnn_out_pool_all, out_pool=args.dagnn_out_pool,
                   dropout=args.dagnn_dropout, num_class=numclass)
    else:
        raise ValueError('Invalid GNN type')

    return model


if __name__ == "__main__":
    main()
