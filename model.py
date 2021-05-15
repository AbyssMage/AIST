import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
from layers import *
from utils import *
import pickle


class AIST(nn.Module):
    def __init__(self, in_hgat, in_fgat, out_gat, att_dot, nhid_rnn, nlayer_rnn, att_rnn, ts, target_region, target_cat, location, nclass=2):
        """
        :param in_hgat: dimension of the input of hgat
        :param in_fgat: dimension of the input of fgat
        :param out_gat: dimension of the output of gat
        :param att_dot: dimension of the dot attention of gat
        :param nhid_rnn: dimension of hidden state of rnn
        :param nlayer_rnn: number of layers of rnn
        :param att_rnn: dimension of attention of trend
        :param att_rnn: number of time steps
        :param target_region: starts with 0
        :param target_cat: starts with 0
        """
        super(AIST, self).__init__()
        self.in_hgat = in_hgat
        self.in_fgat = in_fgat
        self.out_gat = out_gat
        self.att_dot = att_dot
        self.nhid_rnn = nhid_rnn
        self.nlayer_rnn = nlayer_rnn
        self.att_rnn = att_rnn
        self.target_cat = target_cat
        self.target_region = target_region

        self.sp_module1 = Spatial_Module(in_hgat, in_fgat, out_gat, att_dot, 0.5, 0.6, 1, ts, target_region, target_cat, location)

        self.sab1 = self_LSTM_sparse_attn_predict(2*out_gat, nhid_rnn, nlayer_rnn, truncate_length=5, top_k=4, attn_every_k=5)
        self.sab2 = self_LSTM_sparse_attn_predict(in_hgat, nhid_rnn, nlayer_rnn, truncate_length=5, top_k=4, attn_every_k=5)
        self.sab3 = self_LSTM_sparse_attn_predict(in_hgat, nhid_rnn, nlayer_rnn, truncate_length=1, top_k=3, attn_every_k=1)

        self.fc1 = nn.Linear(nhid_rnn, 1)
        self.fc2 = nn.Linear(2 * nhid_rnn, nclass)
        self.fc3 = nn.Linear(3 * nhid_rnn, nclass)

        # parameters for trend-attention
        self.wv = nn.Linear(nhid_rnn, self.att_rnn)  # (S, E) x (E, 1) = (S, 1)
        self.wu = nn.Parameter(torch.zeros(size=(42, self.att_rnn)))  # attention of the trends
        nn.init.xavier_uniform_(self.wu.data, gain=1.414)

        self.dropout_layer = nn.Dropout(p=0.2)

    def forward(self, x_crime, x_crime_daily, x_crime_weekly, x_nei, x_ext, x_sides):

        x_crime = self.sp_module1(x_crime, x_nei, x_ext, x_sides)

        x_con, x_con_attn = self.sab1(x_crime)  # x_con = (B, ts)
        x_con = self.dropout_layer(x_con)

        """re_att = open("Heatmap/re_a_" + str(self.target_region) + "_" + str(self.target_cat) + ".txt", 'ab')
        re_att_arr = x_con_attn.view(42, -1).softmax(dim=1).mean(dim=0).view(1, -1).detach().numpy()
        np.savetxt(re_att, re_att_arr, fmt="%f")
        re_att.close()"""

        x_daily, x_daily_attn = self.sab2(x_crime_daily)  # x_daily = (B, 120/6)
        x_daily = self.dropout_layer(x_daily)

        """d_att = open("Heatmap/d_a_" + str(self.target_region) + "_" + str(self.target_cat) + ".txt", 'ab')
        d_att_arr = x_daily_attn.view(42, -1).softmax(dim=1).mean(dim=0).view(1, -1).detach().numpy()
        np.savetxt(d_att, d_att_arr, fmt="%f")
        d_att.close()"""

        x_weekly, x_weekly_attn = self.sab3(x_crime_weekly)  # x_weekly = (B, 120 / 6*7)
        x_weekly = self.dropout_layer(x_weekly)

        """w_att = open("Heatmap/w_a_" + str(self.target_region) + "_" + str(self.target_cat) + ".txt", 'ab')
        w_att_arr = x_weekly_attn.view(42, -1).softmax(dim=1).mean(dim=0).view(1, -1).detach().numpy()
        np.savetxt(w_att, w_att_arr, fmt="%f")
        w_att.close()"""

        # incorporating attention
        x_con = x_con.unsqueeze(1)
        x_daily = x_daily.unsqueeze(1)
        x_weekly = x_weekly.unsqueeze(1)
        x = torch.cat((x_con, x_daily, x_weekly), 1)

        um = torch.tanh(self.wv(x))  # (B, 3, A)
        um = um.transpose(2, 1)  # [B, A, 3]
        wu = self.wu.unsqueeze(1)
        alpha_m = torch.bmm(wu, um)  # [B, 1, 3]
        alpha_m = alpha_m.squeeze(1)  # [B, 3]
        alpha_m = torch.softmax(alpha_m, dim=1)
        attn_trend = alpha_m.detach()

        """t_att = open("Heatmap/t_a_" + str(self.target_region) + "_" + str(self.target_cat) + ".txt", 'ab')
        t_att_arr = alpha_m.view(42, -1).mean(dim=0).view(1, -1).detach().numpy()
        np.savetxt(t_att, t_att_arr, fmt="%f")
        t_att.close()"""

        alpha_m = alpha_m.unsqueeze(1)

        x = torch.bmm(alpha_m, x)
        x = x.squeeze(1)
        x = torch.tanh(self.fc1(x))

        return x, attn_trend


class Spatial_Module(nn.Module):
    def __init__(self, nfeat_hgat, nfeat_fgat, nhid, att_dot, dropout, alpha, nheads, ts, target_region, target_cat, location):
        """
        :param nfeat_hgat: input dimension of hgat
        :param nfeat_fgat: input dimension of fgat
        :param nhid: output dimension of gat
        :param att_dot: dimension of the dot-product attention of gat
        :param dropout:
        :param alpha:
        :param nheads: number of heads of gat
        :param ts: number of time steps
        :param target_region: starts with 0
        """
        super(Spatial_Module, self).__init__()
        self.nfeat_hgat = nfeat_hgat
        self.nfeat_fgat = nfeat_fgat
        self.nhid = nhid
        self.ts = ts
        self.target_region = target_region
        self.target_cat = target_cat
        self.location = location
        self.gat = [GraphAttentionLayer(nfeat_hgat, nfeat_fgat, nhid, att_dot, target_region, target_cat, dropout=dropout, alpha=alpha) for _ in range(ts)]
        for i, g in enumerate(self.gat):
            self.add_module('gat{}'.format(i), g)

    def forward(self, x_crime, x_regions, x_ext, s_crime):
        B = x_crime.shape[0]
        T = x_crime.shape[1]
        tem_x_regions = x_regions.copy()

        reg = gen_neighbor_index_zero_with_target(self.target_region, self.location)
        label = torch.tensor(reg)

        label = label.repeat(T*B, 1)  # (T*B, N)
        label = label.view(label.shape[0] * label.shape[1], 1).long()  # label-shape = (T * B * N, 1)

        x_crime = x_crime.transpose(1, 0)  # shape = (T, B)
        tem_x_regions.append(x_crime)

        N = len(tem_x_regions)  # Num of actual nodes

        feat = torch.stack(tem_x_regions, 2)  # (T, B, N)
        feat = feat.view(feat.shape[0]*feat.shape[1]*feat.shape[2], 1).long()  # (T*B*N, 1)
        feat = torch.cat([label, feat], dim=1)  # (T*B*N, 2) --> (Node Label, features)
        feat = feat.view(T, B*N, 2)

        nfeat = self.nfeat_fgat
        feat_ext = torch.stack(x_ext, 2)
        feat_ext = feat_ext.view(feat_ext.shape[0] * feat_ext.shape[1] * feat_ext.shape[2], -1).long()  # (T*B*N, nfeat)
        feat_ext = torch.cat([label, feat_ext], dim=1)  # (T*B*N, 2)
        feat_ext = feat_ext.view(T, B * N, nfeat+1)
        # feat_ext = feat_ext.view(T, B, N, nfeat + 1)

        crime_side = torch.stack(s_crime, 2)
        crime_side = crime_side.view(crime_side.shape[0]*crime_side.shape[1]*crime_side.shape[2], -1).long()  # (T*B*N, 1)
        crime_side = torch.cat([label, crime_side], dim=1)  # (T*B*N, 2)
        crime_side = crime_side.view(T, B * N, 2)  # (T, B*N, 2)

        spatial_output = []
        j = 0
        for i in range(120-self.ts, 120):
            np.savetxt("data/gat_crime.txt", feat[i], fmt='%d')
            np.savetxt("data/gat_ext.txt", feat_ext[i], fmt='%d')
            np.savetxt("data/gat_side.txt", crime_side[i], fmt='%d')
            adj, features, features_ext, crime_side_features = load_data_GAT()

            out, ext = self.gat[j](features, adj, features_ext, crime_side_features)  # (N, F')(N, N, dv)
            out = out[:, -1, :]
            ext = ext[:, -1, :]
            out = torch.stack((out, ext), dim=2)

            spatial_output.append(out)
            j = j + 1

        spatial_output = torch.stack(spatial_output, 1)
        return spatial_output

