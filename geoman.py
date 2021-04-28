import torch
from torch import nn
from torch.autograd import Variable
from torch.nn import functional as tf
from torch.nn import init
import torch.nn.functional as F


class GeoMAN(nn.Module):
    def __init__(self, hps, N):
        super(GeoMAN, self).__init__()
        self.hps = hps
        self.N = N  # num of regions

        # Global Spatial Attention
        self.tanh = torch.nn.Tanh()
        self.v_g = nn.Parameter(torch.zeros(1, 1))
        # nn.init.xavier_uniform(self.v_g.data)
        self.w_g = nn.Parameter(torch.zeros(1, 2 * self.hps['h_enc']))
        # nn.init.xavier_uniform(self.w_g.data)
        self.u_g = nn.Parameter(torch.zeros(1, 1))
        # nn.init.xavier_uniform(self.u_g.data)
        self.b_g = nn.Parameter(torch.zeros(hps['batch_size'], self.N))
        # nn.init.xavier_uniform(self.b_g.data)

        # Encoder
        self.enc = nn.LSTMCell(2 * self.hps['in_enc'], self.hps['h_enc'], bias=True)

        # Decoder
        self.dec = nn.LSTMCell(self.hps['h_enc'] + self.hps['out_dec'] + self.hps['n_ext'], self.hps['h_dec'], bias=True)

        # Temporal attention
        self.v_d = nn.Parameter(torch.zeros(1, 1))
        # nn.init.xavier_uniform(self.v_d.data)
        self.w_d = nn.Parameter(torch.zeros(1, 2 * self.hps['h_dec']))
        # nn.init.xavier_uniform(self.w_d.data)
        self.u_d = nn.Parameter(torch.zeros(self.hps['h_dec'], 1))
        # nn.init.xavier_uniform(self.u_d.data)
        self.b_d = nn.Parameter(torch.zeros(self.hps['batch_size'], self.hps['ts_enc']))
        # nn.init.xavier_uniform(self.b_d.data)

        # output
        self.v_y = nn.Parameter(torch.zeros(1, 1))
        # nn.init.xavier_uniform(self.v_y.data)
        self.w_m = nn.Parameter(torch.zeros(1, self.hps['h_dec'] + self.hps['h_enc'] + self.hps['out_dec'] + self.hps['n_ext']))
        # nn.init.xavier_uniform(self.w_m.data)
        self.b_m = nn.Parameter(torch.zeros(self.hps['batch_size'], 1))
        # nn.init.xavier_uniform(self.b_m.data)
        self.b_y = nn.Parameter(torch.zeros(self.hps['batch_size'], 1))
        # nn.init.xavier_uniform(self.b_y.data)

    def forward(self, x, X, ex):
        # input: x = (batch_size, ts_enc)

        # encoder with LSTMs
        outputs = []
        h_t = Variable(torch.zeros(x.size(0), self.hps['h_enc'])).float()  # (bs, h_enc)
        c_t = Variable(torch.zeros(x.size(0), self.hps['h_enc'])).float()  # (bs, h_enc)
        input_g = Variable(torch.zeros(x.size(0), self.hps['in_enc'])).float()  # (bs, h_enc)

        for i, input_l in enumerate(x.chunk(x.size(1), dim=1)):
            # input_l = (batch_size, 1, input_size)
            batch_size = x.size(0)
            """input_l = input_l.contiguous().view(input_l.size()[0], input_l.size()[-1])  # input_t = (batch_size, input_size)
            input_t = torch.cat([input_l, input_g], dim=1)
            # print(input_t.shape)

            # Feed encoder with crime data of target region
            h_t, c_t = self.enc(input_t, (h_t, c_t))
            outputs += [h_t]"""

            # Load data for global spatial attention --- needs to generate ts for i+1 timestep not i
            region_t = X[:, i:i + 1, :]
            region_t = torch.transpose(region_t, 2, 1)
            # print(region_t.shape)

            # Calculate the global spatial attention
            g_t = torch.cat([h_t, c_t], dim=1).view(batch_size, 1, -1).repeat(1, self.N, 1)  # (B, 2m) --> (B, N, 2m)
            g_t = torch.matmul(g_t, self.w_g.T)  # (B, N, 1)
            e_t = torch.matmul(region_t, self.u_g.T)  # (B, N, 1)
            g_t = g_t + e_t  # (B, N, 1)
            g_t = g_t.view(batch_size, -1) + self.b_g  # (B, N)
            g_t = self.tanh(g_t).view(batch_size, self.N, -1)  # (B, N, 1)
            # print(g_t.shape)
            # print(self.v_g.T.shape)
            g_t = torch.matmul(g_t, self.v_g.T).view(batch_size, -1)
            # print(g_t.shape)

            b_t = F.softmax(g_t, dim=1).view(batch_size, self.N, 1)  # (B, N)
            b_t = torch.transpose(b_t, 2, 1)
            # print(g_t.shape, region_t.shape)

            global_att = b_t.mean(dim=0)[-1].detach().view(1, -1).numpy()
            # print(global_att)

            # Calculate the input with global spatial attention
            input_g = torch.bmm(b_t, region_t).view(batch_size, -1)

            input_l = input_l.contiguous().view(input_l.size()[0], input_l.size()[-1])  # input_t = (batch_size, input_size)
            input_t = torch.cat([input_l, input_g], dim=1)
            # print(input_t.shape)

            # Feed encoder with crime data of target region
            h_t, c_t = self.enc(input_t, (h_t, c_t))
            outputs += [h_t]

        outputs = torch.stack(outputs, 1)  # outputs = (batch_size, time_step, hidden_size)

        # initial hidden states of decoder
        d_t = Variable(torch.zeros(x.size(0), self.hps['h_dec'])).float()  # (bs, h_dec)
        s_t = Variable(torch.zeros(x.size(0), self.hps['h_dec'])).float()  # (bs, h_dec)
        y_t = Variable(torch.zeros(x.size(0), self.hps['out_dec'])).float()  # (bs, h_dec)
        outputs_dec = []

        for i in range(self.hps['ts_dec']):
            # calculating context vector -- input for decoder
            dec_t = torch.cat([d_t, s_t], dim=1).view(batch_size, 1, -1).repeat(1, self.hps['ts_enc'],
                                                                                1)  # (B, 2n) --> (B, ts_enc, 2n)
            dec_t = torch.matmul(dec_t, self.w_d.T)  # (B, ts_enc, 1)
            u_t = torch.matmul(outputs, self.u_d)  # (B, ts_enc, 1)
            u_t = dec_t + u_t  # (B, ts_enc, 1)
            u_t = u_t.view(batch_size, -1) + self.b_d # (B, ts_enc)
            u_t = self.tanh(u_t).view(batch_size, self.hps['ts_enc'], -1)  # (B, ts_enc, 1)
            u_t = torch.matmul(u_t, self.v_d.T).view(batch_size, -1)  # (B, ts_enc)

            gm_t = F.softmax(u_t, dim=1).view(batch_size, self.hps['ts_enc'], 1)  # (B, ts_enc, 1)
            gm_t = torch.transpose(gm_t, 2, 1)  # (B, 1, ts_enc)

            c_t = torch.bmm(gm_t, outputs).view(batch_size, -1)  # (B, h_enc)

            # concat the input dec with external features + y^i
            input_dec = torch.cat([c_t, y_t, ex], dim=1)  # (B, h_enc + o_dec + n_ex)

            # Feed decoder the input
            d_t, s_t = self.dec(input_dec, (d_t, s_t))
            outputs_dec += [d_t]

            # Genereate y_t
            y_t = torch.matmul(torch.cat([d_t, input_dec], 1), self.w_m.T) + self.b_m  # (B, 1)

            y_t = torch.matmul(y_t, self.v_y.T).view(batch_size, -1) + self.b_y  # (B, 1)

        return y_t


"""r_crime = torch.ones(42, 12)  # (batch_size, ts)
n_crime = torch.ones(42, 12, 5)  # (bs, ts, N)
model = GeoMAN(hps)
model(r_crime, n_crime)"""