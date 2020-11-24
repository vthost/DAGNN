import os

path = os.path.abspath(__file__)
path = path[:path.rindex("/")] + "/../"
PATH = os.path.abspath(path)

# the latter are only default values used in config
DIR_DATA = os.path.join(PATH, 'data')
DIR_RESULTS = os.path.join(PATH, 'results')
DIR_SAVED_MODELS = os.path.join(PATH, 'saved_models')

NA_SUM = "add"
NA_MAX = "max"
NA_GATED_SUM = "gated_sum"
NA_SELF_ATTN_X = "self_attn_x"  # use xs of preds to compute weights, used to aggregate hs of preds
NA_SELF_ATTN_H = "self_attn_h"
NA_ATTN_X = "attn_x"  # use x and xs of preds to compute weights, used to aggregate hs of preds
NA_ATTN_H = "attn_h"  # use x and hs of preds
NA_MATTN_H = "mattn_h"  # use x and hs of preds

P_MEAN = "mean"
P_ADD = "add"  # do not use "sum" so that can be used to call tg pooling function
P_SUM = "sum"  # do not use "sum" so that can be used to call tg pooling function
P_MAX = "max"
P_ATTN = "attn"
EMB_POOLINGS = [P_MEAN, P_MAX, P_SUM]
POOLINGS = [P_MEAN, P_MAX, P_ATTN, P_ADD]

