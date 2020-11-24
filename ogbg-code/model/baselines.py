import torch
import torch.nn as nn
from torch_scatter import scatter_add


# VT we could use zeros anywhere instead of torch.full, was just not sure about eval in beginning but is argmax

class GuessNodeOneToken(nn.Module):

    def __init__(self, vocab2idx, nodeattributes_mapping, max_seq_len):
        super().__init__()
        nodeattributes_mapping = dict(zip(nodeattributes_mapping["attr idx"],nodeattributes_mapping["attr"]))
        self.attridx2vocabidx = {}  # actual vocab
        for i1, attr in nodeattributes_mapping.items():
            if attr not in vocab2idx: continue
            self.attridx2vocabidx[i1] = vocab2idx[attr]

        self.num_vocab = len(vocab2idx)
        self.max_seq_len = max_seq_len

    def forward(self, batched_data):
        device = batched_data.x.device
        # second nodes in batched graphs
        idx = (batched_data.node_depth == 0).nonzero()[:, 0] + 1
        # src = torch.ones(batched_data.num_graphs).unsqueeze(-1).to(device)
        idx = batched_data.x[idx][:, 1:2]
        out = torch.zeros((batched_data.num_graphs, self.num_vocab)).to(device)
        # out = scatter_add(src, idx, dim=-1, out=out)
        # scatter does not work directly since we need to translate indices, so use loop
        for i in range(batched_data.num_graphs):
            if idx[i].item() in self.attridx2vocabidx:
                print(idx[i].item(), )
                out[i, self.attridx2vocabidx[idx[i].item()]] = 1
        for i in range(batched_data.num_graphs):
            print(torch.argmax(out[i]))
        # x = x[:, 0:1]
        # idx = x == 35
        # i = idx.nonzero()
        # x = x[idx]

        pred_list = [out] + [torch.zeros(batched_data.num_graphs, self.num_vocab).to(device)] * (self.max_seq_len-1)
        return pred_list


class GuessTokensByOccurrence(nn.Module):
    # idx2vocab is just fpr printout/analysis
    def __init__(self, vocab2idx, nodeattributes_mapping, max_seq_len, idx2vocab, min_occ=2):
        super().__init__()
        nodeattributes_mapping = dict(zip(nodeattributes_mapping["attr idx"].astype('int32'),nodeattributes_mapping["attr"]))
        self.attridx2vocabidx = {}  # actual vocab
        for i1, attr in nodeattributes_mapping.items():
            if attr not in vocab2idx: continue
            self.attridx2vocabidx[i1] = vocab2idx[attr]

        self.num_vocab = len(vocab2idx)
        self.max_seq_len = max_seq_len
        self.idx2vocab = idx2vocab
        self.min_occ = min_occ

    def forward(self, batched_data):
        x = batched_data.x
        device = x.device
        # second nodes in batched graphs
        idx = (batched_data.node_depth == 0).nonzero()[:, 0]
        pred_list = [torch.full((batched_data.num_graphs, self.num_vocab), -1).to(device) for _ in range(self.max_seq_len)]
        for i in range(batched_data.num_graphs):
                idx1 = x[idx[i]:(idx[i+1] if i < batched_data.num_graphs-1 else x.shape[0])][:, 1]
                for j in range(0, idx1.shape[0]):
                    idx1[j] = self.attridx2vocabidx[idx1[j].item()] if idx1[j].item() in self.attridx2vocabidx else self.num_vocab
                z = torch.zeros(self.num_vocab + 1, dtype=idx1.dtype).to(device)
                z = scatter_add(torch.ones_like(idx1), idx1, dim=-1, out=z)[:-1]
                # print([self.idx2vocab[k.item()] for k in z.nonzero()])
                # print(z[z.nonzero()].tolist())
                # print("-"*10)
                for j in range(self.max_seq_len):
                    tokidx = torch.argmax(z)
                    if tokidx.shape:
                        tokidx = tokidx[0]
                    else:
                        tokidx = tokidx.item()

                    ct = z[tokidx]
                    if ct >= self.min_occ:
                        pred_list[j][i, tokidx] = 1
                    z[tokidx] = 0

        return pred_list


class PerfectModel(nn.Module):
    # idx2vocab is just fpr printout/analysis
    def __init__(self, vocab2idx, nodeattributes_mapping, max_seq_len, idx2vocab):
        super().__init__()

        nodeattributes_mapping = dict(zip(nodeattributes_mapping["attr idx"].astype('int32'),nodeattributes_mapping["attr"]))
        self.attridx2vocabidx = {}  # actual vocab
        for i1, attr in nodeattributes_mapping.items():
            if attr not in vocab2idx: continue
            self.attridx2vocabidx[i1] = vocab2idx[attr]
        self.num_vocab = len(vocab2idx)
        self.max_seq_len = max_seq_len
        self.idx2vocab = idx2vocab
        self.vocab2idx = vocab2idx

    def forward(self, batched_data):
        x = batched_data.x
        device = x.device
        # second nodes in batched graphs
        idx = (batched_data.node_depth == 0).nonzero()[:, 0]
        pred_list = [torch.full((batched_data.num_graphs, self.num_vocab), -1).to(device) for _ in range(self.max_seq_len)]
        for i in range(batched_data.num_graphs):
                x1 = x[idx[i]:(idx[i+1] if i < batched_data.num_graphs-1 else x.shape[0])][:, 1]
                for j in range(0, x1.shape[0]):
                    x1[j] = self.attridx2vocabidx[x1[j].item()] if x1[j].item() in self.attridx2vocabidx else self.num_vocab

                y = [self.vocab2idx[tok] for tok in batched_data.y[i] if tok in self.vocab2idx]
                for j in range(min(self.max_seq_len, len(y))):
                    if (x1 == y[j]).nonzero().shape[0]:
                        pred_list[j][i, y[j]] = 1

                # print([torch.argmax(pred_list[j][i]).item() for j in range(min(self.max_seq_len, len(y))) if pred_list[j][i][torch.argmax(pred_list[j][i]).item()] >0])
                # print(y)
                # print("-"*10)

        return pred_list

class MostPerfectModel(nn.Module):
    # idx2vocab is just fpr printout/analysis
    def __init__(self, vocab2idx, max_seq_len):
        super().__init__()
        self.num_vocab = len(vocab2idx)
        self.max_seq_len = max_seq_len
        self.vocab2idx = vocab2idx

    def forward(self, batched_data):
        x = batched_data.x
        device = x.device

        pred_list = [torch.full((batched_data.num_graphs, self.num_vocab), -1).to(device) for _ in range(self.max_seq_len)]
        for i in range(batched_data.num_graphs):

                y = [self.vocab2idx[tok] for tok in batched_data.y[i] if tok in self.vocab2idx]
                for j in range(min(self.max_seq_len, len(y))):
                    pred_list[j][i, y[j]] = 1

                # print([torch.argmax(pred_list[j][i]).item() for j in range(min(self.max_seq_len, len(y))) if pred_list[j][i][torch.argmax(pred_list[j][i]).item()] >0])
                # print(y)
                # print("-"*10)

        return pred_list


class MajorityPred(nn.Module):
    def __init__(self, num_class, majority=8):
        super().__init__()

        self.num_class = num_class
        self.majority = majority

    def forward(self, batched_data):  # currently running on CPU and getting from ?
        o = torch.zeros(batched_data.num_graphs, self.num_class)
        o[:, self.majority] = 1
        return o
