import math
import torch
from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module
from utils import *
import torch.nn as nn


class GraphConvolution(Module):
    """
    Simple GCN layer, similar to https://arxiv.org/abs/1609.02907
    """

    def __init__(self, in_features, out_features, bias=True):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.FloatTensor(in_features, out_features))
        if bias:
            self.bias = Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input, adj):
        support = torch.matmul(input, self.weight)
        adj = sp.csr_matrix(adj, dtype=np.float32)
        adj = torch.FloatTensor(np.array(adj.todense()))
        output = torch.matmul(adj, support)

        if self.bias is not None:
            return output + self.bias
        else:
            return output

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'


class MVGCN(Module):

    def __init__(self, nhid, in_recent, in_daily, in_weekly, in_ext):
        super(MVGCN, self).__init__()
        self.ts = 120
        self.nhid = nhid
        self.in_recent = in_recent
        self.in_daily = in_daily
        self.in_weekly = in_weekly
        self.in_ext = in_ext

        self.gcn1 = GraphConvolution(in_recent, nhid)
        self.gcn2 = GraphConvolution(in_daily, nhid)
        self.gcn3 = GraphConvolution(in_weekly, nhid)

        self.gcn11 = GraphConvolution(nhid, nhid)
        self.gcn21 = GraphConvolution(nhid, nhid)
        self.gcn31 = GraphConvolution(nhid, nhid)

        # self.w_fc = nn.Linear(in_ext*self.ts, nhid, bias=False)
        self.w_fc = Parameter(torch.FloatTensor(in_ext*self.ts, nhid))
        # nn.init.xavier_uniform_(self.w_fc.data, gain=1)
        nn.init.uniform_(self.w_fc.data, a=0.0, b=0.01)

        # self.w1 = nn.Linear(nhid, bias=False)
        self.w1 = Parameter(torch.FloatTensor(1, nhid))
        # nn.init.xavier_uniform_(self.w1.data, gain=1)
        nn.init.uniform_(self.w1.data, a=0.0, b=0.01)

        # self.w2 = nn.Linear(nhid, bias=False)
        self.w2 = Parameter(torch.FloatTensor(1, nhid))
        # nn.init.xavier_uniform_(self.w2.data, gain=1)
        nn.init.uniform_(self.w2.data, a=0.0, b=0.01)

        # self.w3 = nn.Linear(nhid, bias=False)
        self.w3 = Parameter(torch.FloatTensor(1, nhid))
        # nn.init.xavier_uniform_(self.w3.data, gain=1)
        nn.init.uniform_(self.w3.data, a=0.0, b=0.01)

        # self.w4 = nn.Linear(nhid, bias=False)
        self.w4 = Parameter(torch.FloatTensor(1, nhid))
        # nn.init.xavier_uniform_(self.w4.data, gain=1)
        nn.init.uniform_(self.w4.data, a=0.0, b=0.01)

        self.sigmoid = torch.nn.Sigmoid()

        self.w = Parameter(torch.FloatTensor(nhid, 1))
        # nn.init.xavier_uniform_(self.w.data, gain=1)
        nn.init.uniform_(self.w.data, a=0.0, b=0.01)

        self.b = Parameter(torch.FloatTensor(1))

    def forward(self, adj, x_r, x_d, x_w, x_e):
        y_r = self.gcn1(x_r, adj)
        y_d = self.gcn2(x_d, adj)
        y_w = self.gcn3(x_w, adj)

        # print(y_r.shape)

        """o1 = y_r[:, -1, :]
        o2 = y_d[:, -1, :]
        o3 = y_w[:, -1, :]"""

        y_r = self.gcn11(y_r, adj)
        y_d = self.gcn21(y_d, adj)
        y_w = self.gcn31(y_w, adj)

        o1 = y_r[:, -1, :]
        o2 = y_d[:, -1, :]
        o3 = y_w[:, -1, :]

        y_e = x_e.view(x_e.shape[0], -1)
        o_ext = torch.matmul(y_e, self.w_fc)

        o = o1 * self.w1 + o2 * self.w2 + o3 * self.w3
        o = o + o_ext + self.sigmoid(o_ext) * o
        o = torch.matmul(o, self.w) + self.b
        return o
