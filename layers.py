import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
from torch.nn.parameter import Parameter
import matplotlib.pyplot as plt
import numpy as np
import timeit


class RNN_GRU(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers=1):
        super(RNN_GRU, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.gru = nn.GRUCell(input_size, hidden_size)
        self.out = nn.Linear(hidden_size, 2)
        self.dropout_layer = nn.Dropout(p=0.2)

    def forward(self, x):
        timestep = 120
        outputs = []
        h_t = Variable(torch.zeros(x.size(0), self.hidden_size)).float()
        for i, input_t in enumerate(x.chunk(timestep, dim=1)):
            # input_t = (batch_size, 1, input_size)
            input_t = input_t.contiguous().view(input_t.size()[0], input_t.size()[-1])  # input_t = (batch_size, 1,
            # input_size) ---> (batch_size, input_size)
            h_t = self.gru(input_t, h_t)  # h_t = (batch_size, hidden_size)
            outputs += [h_t]

        outputs = torch.stack(outputs, 1)  # outputs = (batch_size, time_step, hidden_size)
        # h_t = self.dropout_layer(h_t)
        # h_t = self.out(h_t)
        return outputs, h_t


class RNN_LSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers=1):
        super(RNN_LSTM, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.lstm = nn.LSTMCell(input_size, hidden_size)
        self.out = nn.Linear(hidden_size, 2)
        self.dropout_layer = nn.Dropout(p=0.2)

    def forward(self, x):
        x = x.view(x.shape[0], -1, self.input_size)
        outputs = []
        h_t = Variable(torch.zeros(x.size(0), self.hidden_size)).float()
        c_t = Variable(torch.zeros(x.size(0), self.hidden_size)).float()
        for i, input_t in enumerate(x.chunk(x.size(1), dim=1)):
            # input_t = (ba
            # tch_size, 1, input_size)
            input_t = input_t.contiguous().view(input_t.size()[0], input_t.size()[-1])  # input_t = (batch_size, input_size)
            h_t, c_t = self.lstm(input_t, (h_t, c_t))
            outputs += [h_t]
        outputs = torch.stack(outputs, 1)  # outputs = (batch_size, time_step, hidden_size)
        # h_t = self.dropout_layer(h_t)
        # h_t = self.out(h_t)
        # return outputs, h_t
        return h_t


class Att_RNN_GRU(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers=1):
        super(Att_RNN_GRU, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.att_dim = 40

        self.gru = nn.GRUCell(input_size, hidden_size)
        self.hidden2out = nn.Linear(hidden_size, 1)
        self.dropout_layer = nn.Dropout(p=0.2)

        self.wv = nn.Linear(hidden_size, self.att_dim)  # (S, E) x (E, 1) = (S, 1)
        self.wu = nn.Parameter(torch.rand(self.att_dim))

    def forward(self, x):
        outputs = []
        h_t = Variable(torch.zeros(x.size(0), self.hidden_size)).float()
        for i, input_t in enumerate(x.chunk(x.size(1), dim=1)):
            # input_t = (batch_size, 1, input_size)
            input_t = input_t.contiguous().view(input_t.size()[0], input_t.size()[-1])  # input_t = (batch_size, 1, input_size) ---> (batch_size, input_size)
            h_t = self.gru(input_t, h_t)  # h_t = (batch_size, hidden_size)
            outputs += [h_t]

        outputs = torch.stack(outputs, 1)  # outputs = (batch_size, time_step, hidden_size)
        # h_t = self.dropout_layer(h_t)  # context vector for plain-rnn
        # h_t = self.hidden2out(h_t)

        um = torch.tanh(self.wv(outputs))  # (B, T, H)
        um = um.transpose(2, 1)  # [B*H*T]

        wu = self.wu.repeat(outputs.shape[0], 1).unsqueeze(1)  # [B*1*H]
        alpha_m = torch.bmm(wu, um)  # [B*1*T]
        alpha_m = alpha_m.squeeze(1)  # [B*T]
        alpha_m = torch.softmax(alpha_m, dim=1).unsqueeze(1)
        # print(alpha_m)

        context = torch.bmm(alpha_m, outputs)
        context = self.dropout_layer(context)
        context = self.hidden2out(context)
        context = context.view(context.shape[0], context.shape[1] * context.shape[2])
        return context


class RNN_GRU_Naive_Contextual(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers=1):
        super(RNN_GRU_Naive_Contextual, self).__init__()
        self.embedding_size = 32
        self.linear = nn.Linear(input_size, self.embedding_size)
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.gru = nn.GRUCell(self.embedding_size, hidden_size)
        self.hidden2out = nn.Linear(hidden_size, 2)
        self.dropout_layer = nn.Dropout(p=0.2)

    def forward(self, x):
        x = self.linear(x)
        outputs = []
        h_t = Variable(torch.zeros(x.size(0), self.hidden_size)).float()
        for i, input_t in enumerate(x.chunk(x.size(1), dim=1)):
            # input_t = (batch_size, 1, input_size)
            input_t = input_t.contiguous().view(input_t.size()[0], input_t.size()[-1])  # input_t = (batch_size, 1, input_size) ---> (batch_size, input_size)
            h_t = self.gru(input_t, h_t)  # h_t = (batch_size, hidden_size)
            outputs += [h_t]

        outputs = torch.stack(outputs, 1)  # outputs = (batch_size, time_step, hidden_size)
        h_t = self.dropout_layer(h_t)
        h_t = self.hidden2out(h_t)
        return h_t


class Attention(nn.Module):
    def __init__(self, input_size, output_size, num_layers=1):
        super(Attention, self).__init__()
        self.wv = nn.Linear(input_size, output_size)  # (S, E) x (E, 1) = (S, 1)
        self.wu = nn.Linear(output_size, 1, bias=False)

    def forward(self, x):
        um = torch.tanh(self.wv(x))  # (batch_size, time-step, output-size)
        attention = self.wu(um).squeeze(2)  # (batch_size, time-step, 1) ----> (batch_size, time-step)
        attention = F.softmax(attention, dim=1)
        return attention


class Hierachical_Recurrent_Framework(nn.Module):
    def __init__(self, input_size_crime, input_size_anomaly, hidden_size, num_layers=1):
        super(Hierachical_Recurrent_Framework, self).__init__()
        self.input_size_crime = input_size_crime
        self.input_size_anomaly = input_size_anomaly
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.gru_crime = RNN_GRU(input_size_crime, hidden_size)
        self.gru_anomaly = RNN_GRU(input_size_anomaly, hidden_size)
        self.gru_inter = RNN_GRU(2 * hidden_size, hidden_size)

        self.out = nn.Linear(hidden_size, 2)
        self.dropout_layer = nn.Dropout(p=0.2)
        self.att_dim = 40

        self.wv = nn.Linear(hidden_size, self.att_dim)  # (S, E) x (E, 1) = (S, 1)

        # using as a parameter
        self.wu = nn.Parameter(torch.zeros(self.att_dim))

    def forward(self, crime_input, anomaly_input):
        # outputs = []
        crime_input = self.dropout_layer(crime_input)
        anomaly_input = self.dropout_layer(anomaly_input)
        output_crime, h_crime = self.gru_crime(crime_input)
        output_anomaly, h_anomaly = self.gru_anomaly(anomaly_input)

        input_inter = torch.cat([output_crime, output_anomaly], dim=2)
        output_inter, h_inter = self.gru_inter(input_inter)

        # h_inter = self.dropout_layer(h_inter)
        # h_inter = self.hidden2out(h_inter)

        um = torch.tanh(self.wv(output_inter))  # (B, T, A)
        um = um.transpose(2, 1)  # [B*A*T]

        wu = self.wu.repeat(output_inter.shape[0], 1).unsqueeze(1)  # [B*1*A]

        alpha_m = torch.bmm(wu, um)  # [B*1*T]
        alpha_m = alpha_m.squeeze(1)  # [B*T]
        alpha_m = torch.softmax(alpha_m, dim=1).unsqueeze(1)

        context = torch.bmm(alpha_m, output_inter)
        # context = self.out(context)
        # context = context.view(context.shape[0], context.shape[1] * context.shape[2])
        return context


class MLP(nn.Module):
    def __init__(self, input_size, nhid_mlp, num_layers=1):  # nhid = 3
        super(MLP, self).__init__()
        self.nhid = 40
        self.input = nn.Linear(input_size, self.nhid)
        self.h1 = nn.Linear(self.nhid, self.nhid)
        self.h2 = nn.Linear(self.nhid, self.nhid)
        self.h3 = nn.Linear(self.nhid, self.nhid)
        self.output = nn.Linear(self.nhid, 1)  # changed 2 -> 1
        self.tanh = torch.nn.Tanh()

    def forward(self, x):
        """x = torch.relu(self.input(x))
        x = torch.relu(self.h1(x))
        x = torch.relu(self.h2(x))
        x = torch.relu(self.h3(x))
        x = torch.sigmoid(self.output(x))
        x = x.view(x.shape[0], x.shape[1] * x.shape[2])"""
        x = torch.relu(self.input(x))
        x = torch.relu(self.h1(x))
        x = torch.relu(self.h2(x))
        x = torch.relu(self.h3(x))
        x = self.tanh(self.output(x))
        x = x.view(x.shape[0], x.shape[1] * x.shape[2])
        return x


class MLP_New(nn.Module):
    def __init__(self, input_size, nhid, num_layers=1):  # nhid = 3
        super(MLP_New, self).__init__()
        self.nhid = nhid
        self.input = nn.Linear(input_size, self.nhid)
        self.h1 = nn.Linear(self.nhid, self.nhid)
        self.h2 = nn.Linear(self.nhid, self.nhid)
        self.h3 = nn.Linear(self.nhid, self.nhid)
        self.output = nn.Linear(self.nhid, 2)

    def forward(self, x):
        x = self.input(x)
        x = self.h1(x)
        x = self.h2(x)
        x = self.h3(x)
        x = self.output(x)
        # x = x.view(x.shape[0], x.shape[1] * x.shape[2])
        return x


class Sparse_attention(nn.Module):
    def __init__(self, top_k=5):
        super(Sparse_attention, self).__init__()
        self.top_k = top_k

    def forward(self, attn_s):
        # print("Inside sparsifying------->", attn_s.size())
        # normalize the attention weights using piece-wise Linear function
        # only top k should
        attn_plot = []
        # torch.max() returns both value and location
        # attn_s_max = torch.max(attn_s, dim=1)[0]
        # print(attn_s_max.size())
        # attn_w = torch.clamp(attn_s_max, min=0, max=attn_s_max)
        eps = 10e-8
        batch_size = attn_s.size()[0]
        time_step = attn_s.size()[1]
        if time_step <= self.top_k:
            # just make everything greater than 0, and return it
            # delta = torch.min(attn_s, dim=1)[0]
            """attn_w = torch.clamp(attn_s, min=0)
            attn_w_sum = torch.sum(attn_w, dim=1)
            attn_w_sum = attn_w_sum + eps
            attn_w_normalize = attn_w / attn_w_sum.reshape((batch_size, 1)).repeat(1, time_step)
            return attn_w_normalize"""
            return attn_s
        else:
            # get top k and return it
            # bottom_k = attn_s.size()[1] - self.top_k
            # value of the top k elements
            # delta = torch.kthvalue(attn_s, bottom_k, dim=1)[0]
            delta = torch.topk(attn_s, self.top_k, dim=1)[0][:, -1] + eps  # updated myself
            # return delta
            # delta = torch.topk(attn_s, self.top_k, dim=1)[0][:, -1] + eps
            # delta = attn_s_max - torch.topk(attn_s, self.top_k, dim= 1)[0][:,-1] + eps
            # normalize

        # attn_w = attn_s - delta.repeat(1, time_step)
        attn_w = attn_s - delta.reshape((batch_size, 1)).repeat(1, time_step)
        attn_w = torch.clamp(attn_w, min=0)
        attn_w_sum = torch.sum(attn_w, dim=1)
        attn_w_sum = attn_w_sum + eps
        attn_w_normalize = attn_w / attn_w_sum.reshape((batch_size, 1)).repeat(1, time_step)
        # print("Entered Here")
        return attn_w_normalize


def attention_visualize(attention_timestep, filename):
    # visualize attention
    plt.matshow(attention_timestep)
    filename += '_attention.png'
    plt.savefig(filename)


class self_LSTM_sparse_attn_predict_eval_interpre(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes, param,
                 truncate_length=100, predict_m=10, block_attn_grad_past=False, attn_every_k=1, top_k=5):
        # latest sparse attentive back-prop implementation
        super(self_LSTM_sparse_attn_predict_eval_interpre, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.num_classes = num_classes
        self.block_attn_grad_past = block_attn_grad_past
        self.truncate_length = truncate_length
        # self.lstm1 = nn.LSTMCell(input_size, hidden_size) no need

        self.w = nn.Parameter(torch.zeros(input_size, hidden_size))
        nn.init.xavier_uniform_(self.w.data, gain=1.414)

        self.fc = nn.Linear(hidden_size * 2, num_classes)  # num_classes -> 1
        self.fc1 = nn.Linear(hidden_size, num_classes)
        self.fc2 = nn.Linear(hidden_size, num_classes)
        self.attn_every_k = attn_every_k
        self.top_k = top_k
        self.tanh = torch.nn.Tanh()
        # self.w_t = nn.Parameter(normal(torch.Tensor(self.hidden_size * 2, 1), mean=0.0, std=0.01))  #

        # self.w_t = nn.Parameter(torch.zeros(self.hidden_size * 2, 1))

        self.w_t = nn.Parameter(torch.zeros(self.hidden_size * 2, 1))
        nn.init.xavier_uniform_(self.w_t.data, gain=1.414)
        # self.w_t.requires_grad = False  # for interpretability

        """self.w_t = param
        self.w_t.requires_grad = False  # for interpretability"""

        self.sparse_attn = Sparse_attention(top_k=self.top_k)
        self.predict_m = nn.Linear(hidden_size, 2)  # hidden_size

    def forward(self, x):
        # x = x.view(x.shape[0], int(x.shape[1]/self.input_size), self.input_size)
        batch_size = x.size(0)
        time_size = x.size(1)
        input_size = self.input_size
        hidden_size = self.hidden_size

        h_t = Variable(torch.zeros(batch_size, hidden_size))  # h_t = (batch_size, hidden_size)
        c_t = Variable(torch.zeros(batch_size, hidden_size))  # c_t = (batch_size, hidden_size)
        predict_h = Variable(torch.zeros(batch_size, hidden_size))  # predict_h = (batch_size, hidden_size)

        # Will eventually grow to (batch_size, time_size, hidden_size)
        # with more and more concatenations.
        h_old = h_t.view(batch_size, 1, hidden_size)  # h_old = (batch_size, 1, hidden_size) --> Memory

        outputs = []
        attn_all = []
        attn_w_viz = []
        predicted_all = []
        outputs_new = []

        for i, input_t in enumerate(x.chunk(time_size, dim=1)):
            # input_t = (batch_size, 1, input_size)
            # print("time-step------------------------------------------", i)
            remember_size = h_old.size(1)
            # print("remember_size = ", remember_size)

            if (i + 1) % self.truncate_length == 0:
                h_t, c_t = h_t.detach(), c_t.detach()

            # Feed MLP cell
            input_t = input_t.contiguous().view(batch_size, input_size)  # input_t = (batch_size, input_size)
            # h_t, c_t = self.lstm1(input_t, (h_t, c_t))  # h_t/ c_t = (batch_size, hidden dimension)
            h_t = self.tanh(torch.matmul(input_t, self.w))
            predict_h = self.predict_m(h_t.detach())  # predict_h = (batch_size, hidden dimension) h_t ----> predict_h
            predicted_all.append(h_t)  # changed predict_h

            # Broadcast and concatenate current hidden state against old states
            h_repeated = h_t.unsqueeze(1).repeat(1, remember_size, 1)  # h_repeated = (batch_size, remember_size = memory, hidden_size)
            mlp_h_attn = torch.cat((h_repeated, h_old), 2)  # mlp_h_attn = (batch_size, remember_size, 2* hidden_size)

            if self.block_attn_grad_past:
                mlp_h_attn = mlp_h_attn.detach()

            #
            # Feed the concatenation to the MLP.
            # The tensor shapes being multiplied are
            #     mlp_h_attn.size() = (batch_size, remember_size, 2*hidden_size)
            # by
            #     self.w_t.size()   = (2*hidden_size, 1)
            # Desired result is
            #     attn_w.size()     = (batch_size, remember_size, 1)
            #
            mlp_h_attn = self.tanh(mlp_h_attn)  # mlp_h_attn = (batch_size, remember_size, 2* hidden_size)

            if False:  # PyTorch 0.2.0
                attn_w = torch.matmul(mlp_h_attn, self.w_t)
            else:  # PyTorch 0.1.12
                mlp_h_attn = mlp_h_attn.view(batch_size * remember_size, 2 * hidden_size)  # mlp_h_attn = (batch_size * remember_size, 2* hidden_size)
                attn_w = torch.mm(mlp_h_attn, self.w_t)  # attn_w = (batch_size * remember_size, 1)
                attn_w = attn_w.view(batch_size, remember_size, 1)  # attn_w = (batch_size, remember_size, 1)
            #
            # For each batch example, "select" top-k elements by sparsifying
            # attn_w.size() = (batch_size, remember_size, 1). The top k elements
            # are left non-zero and the other ones are zeroed.
            #
            attn_w = attn_w.view(batch_size, remember_size)  # attn_w = (batch_size, remember_size)
            attn_w = self.sparse_attn(attn_w)  # attn_w = (batch_size, remember_size)
            attn_w = attn_w.view(batch_size, remember_size, 1)  # attn_w = (batch_size, remember_size, 1)

            # if i >= 100:
            # print(attn_w.mean(dim=0).view(remember_size))
            attn_w_viz.append(attn_w.mean(dim=0).view(remember_size))  # you should return it

            #
            # Broadcast the weights against the past remembered hidden states,
            # then compute the attention information attn_c.
            #
            attn_w = attn_w.repeat(1, 1, hidden_size)  # attn_w = (batch_size, remember_size, hidden_size)
            h_old_w = attn_w * h_old  # attn_w = (batch_size, remember_size, hidden_size)
            attn_c = torch.sum(h_old_w, 1).squeeze(1)  # att_c = (batch_size, hidden_size)

            # Feed attn_c to hidden state h_t
            h_t = h_t + attn_c  # h_t = (batch_size, hidden_size)

            #
            # At regular intervals, remember a hidden state, store it in memory
            #
            if (i + 1) % self.attn_every_k == 0:
                h_old = torch.cat((h_old, h_t.view(batch_size, 1, hidden_size)), dim=1)

            predict_real_h_t = self.predict_m(h_t.detach())  # predict_h = (batch_size, hidden dimension) h_t ----> predict_h
            outputs_new += [predict_real_h_t]

            # Record outputs
            outputs += [h_t]

            # For visualization purposes:
            attn_all += [attn_c]

        #
        # Compute return values. These should be:
        #     out        = (batch_size, time_size, num_classes)
        #     attn_w_viz = len([(remember_size)]) == time_size-100
        #
        # print(attn_w)
        predicted_all = torch.stack(predicted_all, 1)  # predicted_all = (batch_size, time_step, hidden_size)
        outputs = torch.stack(outputs, 1)  # outputs = (batch_size, time_step, hidden_size)
        attn_all = torch.stack(attn_all, 1)  # attn_all = (batch_size, time_step, hidden_size)

        # h_outs = outputs.detach()  # h_outs = (batch_size, time_step, hidden_size)
        # outputs = torch.cat((outputs, attn_all), 2)  # outputs = (batch_size, time_step, 2 * hidden_size)

        # shp = outputs.size()
        # out = outputs.contiguous().view(shp[0] * shp[1], shp[2])  # out = (batch_size * time_step, 2 * hidden_size)
        # out = self.fc(out)  # out = (batch_size * time_step, num_classes)
        # out = out.view(shp[0], shp[1], self.num_classes)  # out = (batch_size, time_step, num_classes)
        # out = out[:, -1:, :]
        # out = out.view(shp[0], -1, self.num_classes)  # out = (batch_size, time_step, num_classes)
        # return out, attn_w_viz, predicted_all, h_outs
        # return out
        # h_predict = torch.cat((h_t, attn_c), 1)
        # h_predict = self.fc1(h_t)

        # print(len(attn_w_viz))
        return attn_c


class self_LSTM_sparse_attn_predict(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes,
                 truncate_length=100, predict_m=10, block_attn_grad_past=False, attn_every_k=1, top_k=5):
        # latest sparse attentive back-prop implementation
        super(self_LSTM_sparse_attn_predict, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.num_classes = num_classes
        self.block_attn_grad_past = block_attn_grad_past
        self.truncate_length = truncate_length
        self.lstm1 = nn.LSTMCell(input_size, hidden_size)
        self.fc = nn.Linear(hidden_size * 2, num_classes)  # num_classes -> 1
        self.fc1 = nn.Linear(hidden_size, num_classes)
        self.fc2 = nn.Linear(hidden_size, num_classes)
        self.attn_every_k = attn_every_k
        self.top_k = top_k
        self.tanh = torch.nn.Tanh()
        # self.w_t = nn.Parameter(normal(torch.Tensor(self.hidden_size * 2, 1), mean=0.0, std=0.01))  #

        # self.w_t = nn.Parameter(torch.zeros(self.hidden_size * 2, 1))

        self.w_t = nn.Parameter(torch.zeros(self.hidden_size * 2, 1))
        nn.init.xavier_uniform_(self.w_t.data, gain=1.414)
        # self.w_t.requires_grad = False  # for interpretability

        self.sparse_attn = Sparse_attention(top_k=self.top_k)
        self.predict_m = nn.Linear(hidden_size, 2)  # hidden_size

    def forward(self, x):
        # x = x.view(x.shape[0], int(x.shape[1]/self.input_size), self.input_size)
        batch_size = x.size(0)
        time_size = x.size(1)
        input_size = self.input_size
        hidden_size = self.hidden_size

        h_t = Variable(torch.zeros(batch_size, hidden_size))  # h_t = (batch_size, hidden_size)
        c_t = Variable(torch.zeros(batch_size, hidden_size))  # c_t = (batch_size, hidden_size)
        predict_h = Variable(torch.zeros(batch_size, hidden_size))  # predict_h = (batch_size, hidden_size)

        # Will eventually grow to (batch_size, time_size, hidden_size)
        # with more and more concatenations.
        h_old = h_t.view(batch_size, 1, hidden_size)  # h_old = (batch_size, 1, hidden_size) --> Memory

        outputs = []
        attn_all = []
        attn_w_viz = []
        predicted_all = []
        outputs_new = []

        for i, input_t in enumerate(x.chunk(time_size, dim=1)):
            # input_t = (batch_size, 1, input_size)
            # print("time-step------------------------------------------", i)
            remember_size = h_old.size(1)
            # print("remember_size = ", remember_size)

            if (i + 1) % self.truncate_length == 0:
                h_t, c_t = h_t.detach(), c_t.detach()

            # Feed LSTM Cell
            input_t = input_t.contiguous().view(batch_size, input_size)  # input_t = (batch_size, input_size)
            h_t, c_t = self.lstm1(input_t, (h_t, c_t))  # h_t/ c_t = (batch_size, hidden dimension)
            h_t_naive_lstm = h_t
            predict_h = self.predict_m(h_t.detach())  # predict_h = (batch_size, hidden dimension) h_t ----> predict_h
            predicted_all.append(h_t)  # changed predict_h

            # Broadcast and concatenate current hidden state against old states
            h_repeated = h_t.unsqueeze(1).repeat(1, remember_size, 1)  # h_repeated = (batch_size, remember_size = memory, hidden_size)
            mlp_h_attn = torch.cat((h_repeated, h_old), 2)  # mlp_h_attn = (batch_size, remember_size, 2* hidden_size)

            if self.block_attn_grad_past:
                mlp_h_attn = mlp_h_attn.detach()

            #
            # Feed the concatenation to the MLP.
            # The tensor shapes being multiplied are
            #     mlp_h_attn.size() = (batch_size, remember_size, 2*hidden_size)
            # by
            #     self.w_t.size()   = (2*hidden_size, 1)
            # Desired result is
            #     attn_w.size()     = (batch_size, remember_size, 1)
            #
            mlp_h_attn = self.tanh(mlp_h_attn)  # mlp_h_attn = (batch_size, remember_size, 2* hidden_size)

            if False:  # PyTorch 0.2.0
                attn_w = torch.matmul(mlp_h_attn, self.w_t)
            else:  # PyTorch 0.1.12
                mlp_h_attn = mlp_h_attn.view(batch_size * remember_size, 2 * hidden_size)  # mlp_h_attn = (batch_size * remember_size, 2* hidden_size)
                attn_w = torch.mm(mlp_h_attn, self.w_t)  # attn_w = (batch_size * remember_size, 1)
                attn_w = attn_w.view(batch_size, remember_size, 1)  # attn_w = (batch_size, remember_size, 1)
            #
            # For each batch example, "select" top-k elements by sparsifying
            # attn_w.size() = (batch_size, remember_size, 1). The top k elements
            # are left non-zero and the other ones are zeroed.
            #
            attn_w = attn_w.view(batch_size, remember_size)  # attn_w = (batch_size, remember_size)
            attn_w = self.sparse_attn(attn_w)  # attn_w = (batch_size, remember_size)
            attn_w = attn_w.view(batch_size, remember_size, 1)  # attn_w = (batch_size, remember_size, 1)

            # if i >= 100:
            # print(attn_w.mean(dim=0).view(remember_size))
            attn_w_viz.append(attn_w.mean(dim=0).view(remember_size))  # you should return it
            out_attn_w = attn_w
            #
            # Broadcast the weights against the past remembered hidden states,
            # then compute the attention information attn_c.
            #
            attn_w = attn_w.repeat(1, 1, hidden_size)  # attn_w = (batch_size, remember_size, hidden_size)
            h_old_w = attn_w * h_old  # attn_w = (batch_size, remember_size, hidden_size)
            attn_c = torch.sum(h_old_w, 1).squeeze(1)  # att_c = (batch_size, hidden_size)

            # Feed attn_c to hidden state h_t
            h_t = h_t + attn_c  # h_t = (batch_size, hidden_size)

            #
            # At regular intervals, remember a hidden state, store it in memory
            #
            if (i + 1) % self.attn_every_k == 0:
                h_old = torch.cat((h_old, h_t.view(batch_size, 1, hidden_size)), dim=1)

            predict_real_h_t = self.predict_m(h_t.detach())  # predict_h = (batch_size, hidden dimension) h_t ----> predict_h
            outputs_new += [predict_real_h_t]

            # Record outputs
            outputs += [h_t]

            # For visualization purposes:
            attn_all += [attn_c]

        #
        # Compute return values. These should be:
        #     out        = (batch_size, time_size, num_classes)
        #     attn_w_viz = len([(remember_size)]) == time_size-100
        #
        # print(attn_w)
        predicted_all = torch.stack(predicted_all, 1)  # predicted_all = (batch_size, time_step, hidden_size)
        outputs = torch.stack(outputs, 1)  # outputs = (batch_size, time_step, hidden_size)
        attn_all = torch.stack(attn_all, 1)  # attn_all = (batch_size, time_step, hidden_size)

        # h_outs = outputs.detach()  # h_outs = (batch_size, time_step, hidden_size)
        # outputs = torch.cat((outputs, attn_all), 2)  # outputs = (batch_size, time_step, 2 * hidden_size)

        # shp = outputs.size()
        # out = outputs.contiguous().view(shp[0] * shp[1], shp[2])  # out = (batch_size * time_step, 2 * hidden_size)
        # out = self.fc(out)  # out = (batch_size * time_step, num_classes)
        # out = out.view(shp[0], shp[1], self.num_classes)  # out = (batch_size, time_step, num_classes)
        # out = out[:, -1:, :]
        # out = out.view(shp[0], -1, self.num_classes)  # out = (batch_size, time_step, num_classes)
        # return out, attn_w_viz, predicted_all, h_outs
        # return out
        # h_predict = torch.cat((h_t, attn_c), 1)
        # h_predict = self.fc1(h_t)

        # print(len(attn_w_viz))
        # print(out_attn_w.mean(dim=0).detach())
        return attn_c, out_attn_w


class GraphAttentionLayer(nn.Module):

    def __init__(self, in_features, out_features, target_region, target_cat, dropout, alpha, concat=True):
        super(GraphAttentionLayer, self).__init__()
        self.dropout = dropout
        self.in_features = in_features
        self.out_features = out_features
        self.alpha = alpha
        self.concat = concat
        self.target_region = target_region
        self.target_cat = target_cat

        self.W = nn.Parameter(torch.zeros(size=(in_features, out_features)))
        nn.init.xavier_uniform_(self.W.data, gain=1.414)
        self.Wf = nn.Parameter(torch.zeros(size=(in_features, out_features)))
        nn.init.xavier_uniform_(self.Wf.data, gain=1.414)

        self.a = nn.Parameter(torch.zeros(size=(2 * out_features, 1)))
        nn.init.xavier_uniform_(self.a.data, gain=1.414)
        # self.a.requires_grad = False

        self.leakyrelu = nn.LeakyReLU(self.alpha)
        self.WS = nn.Parameter(torch.zeros(size=(in_features, out_features)))
        nn.init.xavier_uniform_(self.WS.data, gain=1.414)

        self.aS = nn.Parameter(torch.zeros(size=(2 * out_features, 1)))
        nn.init.xavier_uniform_(self.aS.data, gain=1.414)
        # self.aS.requires_grad = False

        self.WS1 = nn.Parameter(torch.zeros(size=(out_features, out_features)))
        nn.init.xavier_uniform_(self.WS1.data, gain=1.414)
        self.aS1 = nn.Parameter(torch.zeros(size=(2 * out_features, 1)))
        nn.init.xavier_uniform_(self.aS1.data, gain=1.414)
        # self.aS1.requires_grad = False

        self.att_dim = 40
        self.emb_dim = out_features
        self.nfeat = 12
        # self.embed = nn.Embedding(10000, self.emb_dim)

        ############################################## Commented this for Feature 5:19 AM

        self.WQ = nn.Parameter(torch.zeros(size=(2, self.att_dim)))
        nn.init.xavier_uniform_(self.WQ.data, gain=1.414)
        # self.WQ.requires_grad = False

        self.WK = nn.Parameter(torch.zeros(size=(2, self.att_dim)))
        nn.init.xavier_uniform_(self.WK.data, gain=1.414)
        # self.WK.requires_grad = False

        self.WV = nn.Parameter(torch.zeros(size=(1, out_features)))
        nn.init.xavier_uniform_(self.WV.data, gain=1.414)
        # self.WV.requires_grad = False

        """self.WQ = nn.Linear(2, self.att_dim, bias=False)
        self.WK = nn.Linear(2, self.att_dim, bias=False)
        self.WV = nn.Linear(1, out_features, bias=False)"""

        self.WF = nn.Linear(self.nfeat, out_features, bias=False)  # For other one

        """self.WF = nn.Linear(1, out_features, bias=False)
        self.WQ = nn.Linear(2*out_features, self.att_dim, bias=False)
        self.WK = nn.Linear(2*out_features, self.att_dim, bias=False)
        self.WV = nn.Linear(out_features, out_features, bias=False)"""

    def forward(self, input, adj, ext_input, side_input):
        """input = input.view(42, -1, 1)
        ext_input = ext_input.view(42, 5, -1)
        side_input = side_input.view(42, 5, -1)
        adj = adj.repeat(42, 1, 1)"""

        input = input.view(42, -1, 1)
        ext_input = ext_input.view(42, -1, self.nfeat)  # No of external features = 1
        side_input = side_input.view(42, -1, 1)  # No of crime occurrences per time step = 1
        adj = adj.repeat(42, 1, 1)

        """
            Find the attention vectors for 
            region_wise crime similarity
        """
        # Find the attention vectors for region_wise crime similarity
        h = torch.matmul(input, self.W)  # h = [h_1, h_2, h_3, ... , h_N] * W
        N = h.size()[1]  # N = Number of Nodes (regions)
        a_input = torch.cat([h.repeat(1, 1, N).view(h.shape[0], N * N, -1), h.repeat(1, N, 1)], dim=2).view(h.shape[0], N, -1, 2 * self.out_features)
        e = self.leakyrelu(torch.matmul(a_input, self.a).squeeze(3))
        zero_vec = -9e15 * torch.ones_like(e)  # shape = (B, N, N)
        attention = torch.where(adj > 0, e, zero_vec)  # shape = (B, N, N)

        # attention = F.softmax(attention, dim=2)  # shape = (B, N, N)

        attention = F.dropout(attention, self.dropout, training=self.training)  # shape = (B, N, N)

        # h_prime = torch.matmul(attention, h)  # shape = (B, N, F'1)

        """Code without batch calculation
        h = torch.mm(input, self.W)  # h = [h_1, h_2, h_3, ... , h_N] * W
        N = h.size()[0]  # N = Number of Nodes
        a_input = torch.cat([h.repeat(1, N).view(N * N, -1), h.repeat(N, 1)], dim=1).view(N, -1, 2 * self.out_features)
        e = self.leakyrelu(torch.matmul(a_input, self.a).squeeze(2))
        zero_vec = -9e15 * torch.ones_like(e)  # shape = (N, N)
        attention = torch.where(adj > 0, e, zero_vec)  # shape = (N, N)
        attention = F.dropout(attention, self.dropout, training=self.training)  # shape = (N, N)
        # h_prime = torch.matmul(attention, h)  # shape = (N, F'1)
        """

        # Tensor shapes and co
        # h.repeat(1, 1, N).view(B, N * N, -1) = (B, NxN, F'), h.repeat(N, 1) = (B, NxN, F')
        # cat = (B, NxN, 2F')
        # a_input = (B, N, N, 2F')
        # torch.matmul(a_input, self.a).squeeze(2) = ((B, N, N, 1) -----> (B, N, N))

        """
            Find the attention vectors for 
            side_wise crime similarity
        """
        h_side = torch.matmul(side_input, self.WS)  # h = [h_1, h_2, h_3, ... , h_N] * W
        a_input_side = torch.cat([h_side.repeat(1, 1, N).view(42, N * N, -1), h_side.repeat(1, N, 1)], dim=2).view(42, N, -1, 2 * self.out_features)
        e_side = self.leakyrelu(torch.matmul(a_input_side, self.aS).squeeze(3))
        attention_side = torch.where(adj > 0, e_side, zero_vec)  # shape = (B, N, N)
        attention_side = F.dropout(attention_side, self.dropout, training=self.training)  # shape = (B, N, N)
        # h_prime_side = torch.matmul(attention_side, h_side)  # shape = (B, N, F')

        """
            Find the crime representation of 
            a region
        """

        attention = attention + attention_side
        attention = torch.where(attention > 0, attention, zero_vec)  # shape = (B, N, N)
        attention = F.softmax(attention, dim=2)  # shape = (B, N, N)
        attention = F.dropout(attention, self.dropout, training=self.training)  # shape = (B, N, N)
        h_prime = torch.matmul(attention, h)  # shape = (B, N, F')

        # np.savetxt("attention_region.txt", attention.detach().numpy(), fmt="%.4f")
        # np.savetxt("attention_side.txt", attention_side.detach().numpy(), fmt="%.4f")
        # np.savetxt("attention_total.txt", attention.detach().numpy(), fmt="%.4f")
        # print(attention[:, -1, :].mean(dim=0).detach())

        """r_att = open("Heatmap/r_a_" + str(self.target_region) + "_" + str(self.target_cat) + ".txt", 'ab')
        r_att_arr = attention[:, -1, :].mean(dim=0).view(1, -1).detach().numpy()
        np.savetxt(r_att, r_att_arr, fmt="%f")
        r_att.close()"""

        # Generate Query
        n_feature = 12  # for now poi but should be
        q = torch.cat([input.repeat(1, 1, N).view(input.shape[0], N * N, -1), input.repeat(1, N, 1)], dim=2).view(input.shape[0], N, N, -1)
        q = torch.matmul(q, self.WQ)  # (B, N, N, dq) = (B, N, N, 2) * (2, dq)
        q = q / (self.att_dim ** 0.5)
        q = q.unsqueeze(3)  # (B, N, N, 1, dq)
        # print(q.mean(dim=0))

        # Generate Key
        # hf = self.WF(ext_input.unsqueeze(3))  # (B, N, nfeat, F') =
        ext_input = ext_input.unsqueeze(3)
        k = torch.cat([ext_input.repeat(1, 1, N, 1).view(ext_input.shape[0], N * N, n_feature, -1),
                             ext_input.repeat(1, N, 1, 1).view(ext_input.shape[0], N * N, n_feature, -1)], dim=3).view(ext_input.shape[0], N,
                                                                                                         N, n_feature, 2)
        k = torch.matmul(k, self.WK)  # (B, N, N, nfeat, dk) = (B, N, N, nfeat, 2)* (2, dk)
        k = torch.transpose(k, 4, 3)  # (B, N, N, dk, nfeat)

        # Generate Value
        v = torch.matmul(ext_input, self.WV)  # (B, N, N, nfeat, dv)

        # Generate dot product attention
        dot_attention = torch.matmul(q, k).squeeze(3)  # (B, N, N, nfeat)
        # print(dot_attention[:, -1, :, :].mean(dim=0).detach())
        zero_vec = -9e15 * torch.ones_like(dot_attention)
        dot_attention = torch.where(dot_attention > 0, dot_attention, zero_vec)  # (B, N, N, nfeat)
        dot_attention = F.softmax(dot_attention, dim=3)  # shape = (B, N, N, nfeat)
        # print(dot_attention.mean(dim=0)[-1].sum(dim=0).detach().softmax(dim=0))

        """f_att = open("Heatmap/f_a_" + str(self.target_region) + "_" + str(self.target_cat) + ".txt", 'ab')
        f_att_arr = dot_attention.mean(dim=0)[-1].sum(dim=0).detach().softmax(dim=0).view(1, -1).numpy()
        np.savetxt(f_att, f_att_arr, fmt="%f")
        f_att.close()"""

        """
            Generate the external representation of the regions
        """

        crime_attention = attention.unsqueeze(3).repeat(1, 1, 1, n_feature)
        final_attention = dot_attention * crime_attention
        ext_rep = torch.matmul(final_attention, v)  # shape = (B, N, N, dv)
        ext_rep = ext_rep.sum(dim=2)  # shape = (B, N, N, dv)

        #  hGAT + F
        # ext_rep = self.WF(ext_input)  # (B, N, nfeat, F')

        """
        # Generate Query
        n_feature = 12  # for now poi but should be
        q = self.WQ(a_input)  # (B, N, N, dq) = (B, N, N, 2F') * (2F', dq)
        q = q / (self.att_dim ** 0.5)
        q = q.unsqueeze(4)  # (B, N, N, dq, 1)

        # Generate Key
        hf = self.WF(ext_input.unsqueeze(3))  # (B, N, nfeat, F') =
        f_input = torch.cat([hf.repeat(1, 1, N, 1).view(hf.shape[0], N * N, n_feature, -1), hf.repeat(1, N, 1, 1).view(hf.shape[0], N * N, n_feature, -1)], dim=3).view(hf.shape[0], N, N, n_feature, 2 * self.out_features)
        k = self.WK(f_input)  # (B, N, N, nfeat, dk) = (B, N, N, nfeat, 2F')* (2F', dk)
        # k = torch.transpose(k, 4, 3)  # (B, N, N, dk, nfeat)

        # Generate Value
        v = self.WV(hf)  # (B, N, N, nfeat, dv)

        # Generate dot product attention
        dot_attention = torch.matmul(k, q).squeeze(4)  # (B, N, N, nfeat)
        zero_vec = -9e15 * torch.ones_like(dot_attention)
        dot_attention = torch.where(dot_attention > 0, dot_attention, zero_vec)  # (B, N, N, nfeat)
        dot_attention = F.softmax(dot_attention, dim=3)  # shape = (B, N, N, nfeat)
        print(dot_attention[-1].mean(dim=0).detach())
        # Generate the external representation of the regions
        ext_rep = torch.matmul(dot_attention, v)  # shape = (B, N, N, dv)
        ext_rep = ext_rep.sum(dim=2)  # shape = (B, N, N, dv)
        """

        # final ext_rep(new)
        """reg_attention = attention.view(N*N, 1)
        # print(reg_attention.shape)
        reg_attention = reg_attention.repeat(1, n_feature)
        reg_attention = reg_attention.view(N, N, -1)
        dot_attention = dot_attention.view(N, N, -1)

        total_attention = dot_attention*reg_attention
        total_attention = F.softmax(total_attention, dim=1)  # shape = (N*N, nfeat)
        total_attention = total_attention.view(N, N * n_feature)
        ext_rep = torch.matmul(total_attention, v)  # (N, N, dv)"""

        # ext_rep = torch.zeros((42, self.out_features))

        if self.concat:
            return F.elu(h_prime), F.elu(ext_rep)
        else:
            return h_prime, ext_rep

    def __repr__(self):
        return self.__class__.__name__ + ' (' + str(self.in_features) + ' -> ' + str(self.out_features) + ')'


class GraphAttentionLayer_GAT(nn.Module):

    def __init__(self, in_features, out_features, target_region, target_cat, dropout, alpha, concat=True):
        super(GraphAttentionLayer_GAT, self).__init__()
        self.dropout = dropout
        self.in_features = in_features
        self.out_features = out_features
        self.alpha = alpha
        self.concat = concat
        self.target_region = target_region
        self.target_cat = target_cat

        self.W = nn.Parameter(torch.zeros(size=(in_features, out_features)))
        nn.init.xavier_uniform_(self.W.data, gain=1.414)
        self.Wf = nn.Parameter(torch.zeros(size=(in_features, out_features)))
        nn.init.xavier_uniform_(self.Wf.data, gain=1.414)

        self.a = nn.Parameter(torch.zeros(size=(2 * out_features, 1)))
        nn.init.xavier_uniform_(self.a.data, gain=1.414)
        # self.a.requires_grad = False

        self.leakyrelu = nn.LeakyReLU(self.alpha)
        self.WS = nn.Parameter(torch.zeros(size=(in_features, out_features)))
        nn.init.xavier_uniform_(self.WS.data, gain=1.414)

        self.aS = nn.Parameter(torch.zeros(size=(2 * out_features, 1)))
        nn.init.xavier_uniform_(self.aS.data, gain=1.414)
        # self.aS.requires_grad = False

        self.WS1 = nn.Parameter(torch.zeros(size=(out_features, out_features)))
        nn.init.xavier_uniform_(self.WS1.data, gain=1.414)
        self.aS1 = nn.Parameter(torch.zeros(size=(2 * out_features, 1)))
        nn.init.xavier_uniform_(self.aS1.data, gain=1.414)
        # self.aS1.requires_grad = False

        self.att_dim = 40
        self.emb_dim = out_features
        self.nfeat = 12
        # self.embed = nn.Embedding(10000, self.emb_dim)

        ############################################## Commented this for Feature 5:19 AM

        self.WQ = nn.Parameter(torch.zeros(size=(2, self.att_dim)))
        nn.init.xavier_uniform_(self.WQ.data, gain=1.414)
        # self.WQ.requires_grad = False

        self.WK = nn.Parameter(torch.zeros(size=(2, self.att_dim)))
        nn.init.xavier_uniform_(self.WK.data, gain=1.414)
        # self.WK.requires_grad = False

        self.WV = nn.Parameter(torch.zeros(size=(1, out_features)))
        nn.init.xavier_uniform_(self.WV.data, gain=1.414)
        # self.WV.requires_grad = False

        """self.WQ = nn.Linear(2, self.att_dim, bias=False)
        self.WK = nn.Linear(2, self.att_dim, bias=False)
        self.WV = nn.Linear(1, out_features, bias=False)"""

        self.WF = nn.Linear(self.nfeat, out_features, bias=False)  # For other one

        """self.WF = nn.Linear(1, out_features, bias=False)
        self.WQ = nn.Linear(2*out_features, self.att_dim, bias=False)
        self.WK = nn.Linear(2*out_features, self.att_dim, bias=False)
        self.WV = nn.Linear(out_features, out_features, bias=False)"""

    def forward(self, input, adj, ext_input, side_input):
        """input = input.view(42, -1, 1)
        ext_input = ext_input.view(42, 5, -1)
        side_input = side_input.view(42, 5, -1)
        adj = adj.repeat(42, 1, 1)"""

        input = input.view(42, -1, 1)
        ext_input = ext_input.view(42, -1, self.nfeat)  # No of external features = 1
        side_input = side_input.view(42, -1, 1)  # No of crime occurrences per time step = 1
        adj = adj.repeat(42, 1, 1)

        """
            Find the attention vectors for 
            region_wise crime similarity
        """
        # Find the attention vectors for region_wise crime similarity
        h = torch.matmul(input, self.W)  # h = [h_1, h_2, h_3, ... , h_N] * W
        N = h.size()[1]  # N = Number of Nodes (regions)
        a_input = torch.cat([h.repeat(1, 1, N).view(h.shape[0], N * N, -1), h.repeat(1, N, 1)], dim=2).view(h.shape[0], N, -1, 2 * self.out_features)
        e = self.leakyrelu(torch.matmul(a_input, self.a).squeeze(3))
        zero_vec = -9e15 * torch.ones_like(e)  # shape = (B, N, N)
        attention = torch.where(adj > 0, e, zero_vec)  # shape = (B, N, N)

        # Comment at 10/3 3:17 AM
        attention = F.softmax(attention, dim=2)  # shape = (B, N, N)

        attention = F.dropout(attention, self.dropout, training=self.training)  # shape = (B, N, N)

        # Comment at 10/3 3:17 AM
        h_prime = torch.matmul(attention, h)  # shape = (B, N, F'1)

        """Code without batch calculation
        h = torch.mm(input, self.W)  # h = [h_1, h_2, h_3, ... , h_N] * W
        N = h.size()[0]  # N = Number of Nodes
        a_input = torch.cat([h.repeat(1, N).view(N * N, -1), h.repeat(N, 1)], dim=1).view(N, -1, 2 * self.out_features)
        e = self.leakyrelu(torch.matmul(a_input, self.a).squeeze(2))
        zero_vec = -9e15 * torch.ones_like(e)  # shape = (N, N)
        attention = torch.where(adj > 0, e, zero_vec)  # shape = (N, N)
        attention = F.dropout(attention, self.dropout, training=self.training)  # shape = (N, N)
        # h_prime = torch.matmul(attention, h)  # shape = (N, F'1)
        """

        # Tensor shapes and co
        # h.repeat(1, 1, N).view(B, N * N, -1) = (B, NxN, F'), h.repeat(N, 1) = (B, NxN, F')
        # cat = (B, NxN, 2F')
        # a_input = (B, N, N, 2F')
        # torch.matmul(a_input, self.a).squeeze(2) = ((B, N, N, 1) -----> (B, N, N))

        """
            Find the attention vectors for 
            side_wise crime similarity
        """
        """h_side = torch.matmul(side_input, self.WS)  # h = [h_1, h_2, h_3, ... , h_N] * W
        a_input_side = torch.cat([h_side.repeat(1, 1, N).view(42, N * N, -1), h_side.repeat(1, N, 1)], dim=2).view(42, N, -1, 2 * self.out_features)
        e_side = self.leakyrelu(torch.matmul(a_input_side, self.aS).squeeze(3))
        attention_side = torch.where(adj > 0, e_side, zero_vec)  # shape = (B, N, N)
        attention_side = F.dropout(attention_side, self.dropout, training=self.training)"""  # shape = (B, N, N)
        # h_prime_side = torch.matmul(attention_side, h_side)  # shape = (B, N, F')

        """
            Find the crime representation of 
            a region
        """

        """attention = attention + attention_side
        attention = torch.where(attention > 0, attention, zero_vec)  # shape = (B, N, N)
        attention = F.softmax(attention, dim=2)  # shape = (B, N, N)
        attention = F.dropout(attention, self.dropout, training=self.training)  # shape = (B, N, N)
        h_prime = torch.matmul(attention, h)"""  # shape = (B, N, F')

        # np.savetxt("attention_region.txt", attention.detach().numpy(), fmt="%.4f")
        # np.savetxt("attention_side.txt", attention_side.detach().numpy(), fmt="%.4f")
        # np.savetxt("attention_total.txt", attention.detach().numpy(), fmt="%.4f")
        # print(attention[:, -1, :].mean(dim=0).detach())

        """r_att = open("Heatmap/r_a_" + str(self.target_region) + "_" + str(self.target_cat) + ".txt", 'ab')
        r_att_arr = attention[:, -1, :].mean(dim=0).view(1, -1).detach().numpy()
        np.savetxt(r_att, r_att_arr, fmt="%f")
        r_att.close()"""

        # Generate Query
        n_feature = 12  # for now poi but should be
        q = torch.cat([input.repeat(1, 1, N).view(input.shape[0], N * N, -1), input.repeat(1, N, 1)], dim=2).view(input.shape[0], N, N, -1)
        q = torch.matmul(q, self.WQ)  # (B, N, N, dq) = (B, N, N, 2) * (2, dq)
        q = q / (self.att_dim ** 0.5)
        q = q.unsqueeze(3)  # (B, N, N, 1, dq)
        # print(q.mean(dim=0))

        # Generate Key
        # hf = self.WF(ext_input.unsqueeze(3))  # (B, N, nfeat, F') =
        ext_input = ext_input.unsqueeze(3)
        k = torch.cat([ext_input.repeat(1, 1, N, 1).view(ext_input.shape[0], N * N, n_feature, -1),
                             ext_input.repeat(1, N, 1, 1).view(ext_input.shape[0], N * N, n_feature, -1)], dim=3).view(ext_input.shape[0], N,
                                                                                                         N, n_feature, 2)
        k = torch.matmul(k, self.WK)  # (B, N, N, nfeat, dk) = (B, N, N, nfeat, 2)* (2, dk)
        k = torch.transpose(k, 4, 3)  # (B, N, N, dk, nfeat)

        # Generate Value
        v = torch.matmul(ext_input, self.WV)  # (B, N, N, nfeat, dv)

        # Generate dot product attention
        dot_attention = torch.matmul(q, k).squeeze(3)  # (B, N, N, nfeat)
        # print(dot_attention[:, -1, :, :].mean(dim=0).detach())
        zero_vec = -9e15 * torch.ones_like(dot_attention)
        dot_attention = torch.where(dot_attention > 0, dot_attention, zero_vec)  # (B, N, N, nfeat)
        dot_attention = F.softmax(dot_attention, dim=3)  # shape = (B, N, N, nfeat)"""
        # print(dot_attention.mean(dim=0)[-1].sum(dim=0).detach().softmax(dim=0))

        """f_att = open("Heatmap/f_a_" + str(self.target_region) + "_" + str(self.target_cat) + ".txt", 'ab')
        f_att_arr = dot_attention.mean(dim=0)[-1].sum(dim=0).detach().softmax(dim=0).view(1, -1).numpy()
        np.savetxt(f_att, f_att_arr, fmt="%f")
        f_att.close()"""

        """
            Generate the external representation of the regions
        """

        crime_attention = attention.unsqueeze(3).repeat(1, 1, 1, n_feature)
        final_attention = dot_attention * crime_attention
        ext_rep = torch.matmul(final_attention, v)  # shape = (B, N, N, dv)
        ext_rep = ext_rep.sum(dim=2)  # shape = (B, N, N, dv)

        #  hGAT + F
        # ext_rep = self.WF(ext_input)  # (B, N, nfeat, F')

        """
        # Generate Query
        n_feature = 12  # for now poi but should be
        q = self.WQ(a_input)  # (B, N, N, dq) = (B, N, N, 2F') * (2F', dq)
        q = q / (self.att_dim ** 0.5)
        q = q.unsqueeze(4)  # (B, N, N, dq, 1)

        # Generate Key
        hf = self.WF(ext_input.unsqueeze(3))  # (B, N, nfeat, F') =
        f_input = torch.cat([hf.repeat(1, 1, N, 1).view(hf.shape[0], N * N, n_feature, -1), hf.repeat(1, N, 1, 1).view(hf.shape[0], N * N, n_feature, -1)], dim=3).view(hf.shape[0], N, N, n_feature, 2 * self.out_features)
        k = self.WK(f_input)  # (B, N, N, nfeat, dk) = (B, N, N, nfeat, 2F')* (2F', dk)
        # k = torch.transpose(k, 4, 3)  # (B, N, N, dk, nfeat)

        # Generate Value
        v = self.WV(hf)  # (B, N, N, nfeat, dv)

        # Generate dot product attention
        dot_attention = torch.matmul(k, q).squeeze(4)  # (B, N, N, nfeat)
        zero_vec = -9e15 * torch.ones_like(dot_attention)
        dot_attention = torch.where(dot_attention > 0, dot_attention, zero_vec)  # (B, N, N, nfeat)
        dot_attention = F.softmax(dot_attention, dim=3)  # shape = (B, N, N, nfeat)
        print(dot_attention[-1].mean(dim=0).detach())
        # Generate the external representation of the regions
        ext_rep = torch.matmul(dot_attention, v)  # shape = (B, N, N, dv)
        ext_rep = ext_rep.sum(dim=2)  # shape = (B, N, N, dv)
        """

        # final ext_rep(new)
        """reg_attention = attention.view(N*N, 1)
        # print(reg_attention.shape)
        reg_attention = reg_attention.repeat(1, n_feature)
        reg_attention = reg_attention.view(N, N, -1)
        dot_attention = dot_attention.view(N, N, -1)

        total_attention = dot_attention*reg_attention
        total_attention = F.softmax(total_attention, dim=1)  # shape = (N*N, nfeat)
        total_attention = total_attention.view(N, N * n_feature)
        ext_rep = torch.matmul(total_attention, v)  # (N, N, dv)"""

        # ext_rep = torch.zeros((42, self.out_features))

        if self.concat:
            return F.elu(h_prime), F.elu(ext_rep)
        else:
            return h_prime, ext_rep

    def __repr__(self):
        return self.__class__.__name__ + ' (' + str(self.in_features) + ' -> ' + str(self.out_features) + ')'