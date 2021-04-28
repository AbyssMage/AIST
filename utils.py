import numpy as np
np.set_printoptions(threshold=np.inf)
import scipy.sparse as sp
import torch
from sklearn.metrics import f1_score
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import MinMaxScaler
import scipy


def print_model_parameters(model):
    for name, param in model.named_parameters():
        if param.requires_grad:
            print(name, param.data)


def create_inout_sequences(input_data, tw):
    # input data = (raw data of a region), tw = time-step
    forecast = 1  # Num of days to forecast in the future
    # Consecutive_temporal_data_generation
    in_seq1 = torch.from_numpy(np.ones((8000, tw), dtype=np.int))
    out_seq1 = torch.from_numpy(np.ones((8000, forecast), dtype=np.int))
    L = input_data.shape[0]
    for i in range(L - tw - forecast):
        train_seq = input_data[i:i + tw, :]
        in_seq1[i] = train_seq.view(train_seq.shape[0] * train_seq.shape[1])
        # print(train_seq.view(train_seq.shape[0] * train_seq.shape[1]).np())
        train_label = input_data[i + tw:i + tw + forecast, :]
        out_seq1[i] = train_label.view(train_label.shape[0] * train_label.shape[1])
        # in_seq.append(train_seq)
        # out_seq.append(train_label)
    in_seq1 = in_seq1[:i + 1, :]
    out_seq1 = out_seq1[:i + 1, :]

    # Daily_temporal_data_generation
    batch_size = in_seq1.shape[0]
    time_step_daily = int(tw / 6)
    in_seq2 = torch.from_numpy(np.ones((batch_size, time_step_daily), dtype=np.int))
    out_seq2 = out_seq1
    for i in range(batch_size):
        k = 0
        for j in range(tw):
            if j % 6 == 0:
                in_seq2[i][k] = in_seq1[i][j]
                k = k + 1

    # Weekly_temporal_data_generation
    time_step_weekly = int(tw / (6 * 7)) + 1
    in_seq3 = torch.from_numpy(np.ones((batch_size, time_step_weekly), dtype=np.int))
    out_seq3 = out_seq1
    for i in range(batch_size):
        k = 0
        for j in range(tw):
            if j % (6 * 7) == 0:
                in_seq3[i][k] = in_seq1[i][j]
                k = k + 1
    return in_seq1, out_seq1, in_seq2, in_seq3


def create_inout_sequences_param_Td(input_data, tw):
    # input data = (raw data of a region), tw = time-step
    forecast = 1  # Num of days to forecast in the future
    # Consecutive_temporal_data_generation
    in_seq1 = torch.from_numpy(np.ones((8000, tw), dtype=np.int))
    out_seq1 = torch.from_numpy(np.ones((8000, forecast), dtype=np.int))
    L = input_data.shape[0]
    for i in range(L - tw - forecast):
        train_seq = input_data[i:i + tw, :]
        in_seq1[i] = train_seq.view(train_seq.shape[0] * train_seq.shape[1])
        # print(train_seq.view(train_seq.shape[0] * train_seq.shape[1]).np())
        train_label = input_data[i + tw:i + tw + forecast, :]
        out_seq1[i] = train_label.view(train_label.shape[0] * train_label.shape[1])
        # in_seq.append(train_seq)
        # out_seq.append(train_label)
    in_seq1 = in_seq1[:i + 1, :]
    out_seq1 = out_seq1[:i + 1, :]

    # print(in_seq1.shape)
    inc = 4
    # Daily_temporal_data_generation
    batch_size = in_seq1.shape[0]  # 2190
    time_step_daily = int(tw / 6) + inc  # 20 + inc
    in_seq2 = torch.from_numpy(np.ones((batch_size, time_step_daily), dtype=np.int))
    out_seq2 = out_seq1
    for i in range(batch_size-1, -1, -1):
        k = inc
        for j in range(tw):
            if j % 6 == 0:
                in_seq2[i][k] = in_seq1[i][j]
                k = k + 1

        n = i-1  # copies the sample number
        k = inc - 1
        for m in range(tw):
            if i+1 == 0:
                n = in_seq1.shape[0] - 1
            if m % 6 == 0:
                in_seq2[i][k] = in_seq1[n][m]
                k = k - 1
            if k == -1:
                break

    # Weekly_temporal_data_generation
    time_step_weekly = int(tw / (6 * 7)) + 1
    in_seq3 = torch.from_numpy(np.ones((batch_size, time_step_weekly), dtype=np.int))
    out_seq3 = out_seq1
    for i in range(batch_size):
        k = 0
        for j in range(tw):
            if j % (6 * 7) == 0:
                in_seq3[i][k] = in_seq1[i][j]
                k = k + 1
    return in_seq1, out_seq1, in_seq2, in_seq3


def create_inout_sequences_param_Tw(input_data, tw):
    # input data = (raw data of a region), tw = time-step
    forecast = 1  # Num of days to forecast in the future
    # Consecutive_temporal_data_generation
    in_seq1 = torch.from_numpy(np.ones((8000, tw), dtype=np.int))
    out_seq1 = torch.from_numpy(np.ones((8000, forecast), dtype=np.int))
    L = input_data.shape[0]
    for i in range(L - tw - forecast):
        train_seq = input_data[i:i + tw, :]
        in_seq1[i] = train_seq.view(train_seq.shape[0] * train_seq.shape[1])
        # print(train_seq.view(train_seq.shape[0] * train_seq.shape[1]).np())
        train_label = input_data[i + tw:i + tw + forecast, :]
        out_seq1[i] = train_label.view(train_label.shape[0] * train_label.shape[1])
        # in_seq.append(train_seq)
        # out_seq.append(train_label)
    in_seq1 = in_seq1[:i + 1, :]
    out_seq1 = out_seq1[:i + 1, :]

    # Daily_temporal_data_generation
    batch_size = in_seq1.shape[0]
    time_step_daily = int(tw / 6)
    in_seq2 = torch.from_numpy(np.ones((batch_size, time_step_daily), dtype=np.int))
    out_seq2 = out_seq1
    for i in range(batch_size):
        k = 0
        for j in range(tw):
            if j % 6 == 0:
                in_seq2[i][k] = in_seq1[i][j]
                k = k + 1

    # Weekly_temporal_data_generation
    inc = 3
    time_step_weekly = int(tw / (6 * 7)) + 1 + inc
    in_seq3 = torch.from_numpy(np.ones((batch_size, time_step_weekly), dtype=np.int))
    out_seq3 = out_seq1
    for i in range(batch_size-1, -1, -1):
        k = inc
        for j in range(tw):
            if j % (6 * 7) == 0:
                in_seq3[i][k] = in_seq1[i][j]
                k = k + 1
        k = inc - 1
        n = i - 1  # copies the sample number
        for m in range(tw):
            if i - 1 == 0:
                n = in_seq1.shape[0] - 1
            if m % 6 == 0:
                in_seq2[i][k] = in_seq1[n][m]
                k = k - 1
            if k == -1:
                break
    return in_seq1, out_seq1, in_seq2, in_seq3


def create_inout_sequences_GeoMAN(input_data, tw):
    # input data = (raw data of a region), tw = time-step
    forecast = 1  # Num of days to forecast in the future

    # Consecutive_temporal_data_generation
    in_seq1 = torch.from_numpy(np.ones((8000, tw), dtype=np.int))
    out_seq1 = torch.from_numpy(np.ones((8000, forecast), dtype=np.int))
    L = input_data.shape[0]
    for i in range(L - tw - forecast):
        train_seq = input_data[i:i + tw, :]
        in_seq1[i] = train_seq.view(train_seq.shape[0] * train_seq.shape[1])
        # print(train_seq.view(train_seq.shape[0] * train_seq.shape[1]).np())
        train_label = input_data[i + tw:i + tw + forecast, :]
        out_seq1[i] = train_label.view(train_label.shape[0] * train_label.shape[1])
        # in_seq.append(train_seq)
        # out_seq.append(train_label)
    in_seq1 = in_seq1[:i + 1, :]
    out_seq1 = out_seq1[:i + 1, :]

    return in_seq1, out_seq1


def accuracy(output, labels):
    preds = output.max(1)[1].type_as(labels)
    correct = preds.eq(labels).double()
    correct = correct.sum()
    macro_f1 = f1_score(preds, labels, average='macro')
    micro_f1 = f1_score(preds, labels, average='micro')
    f1 = f1_score(preds, labels)
    # auc = roc_auc_score(preds, labels)
    print(preds)
    print(labels)
    return correct / len(labels), macro_f1, micro_f1, f1


def load_data_GAT():
    # build features
    idx_features_labels = np.genfromtxt("gat_feat.txt", dtype=np.dtype(str))  # (Nodes, NodeLabel+ features + label)
    features = sp.csr_matrix(idx_features_labels[:, 1:], dtype=np.float32)  # (Nodes, features)
    # build features_ext
    idx_features_labels_ext = np.genfromtxt("gat_feat_ext.txt",
                                            dtype=np.dtype(str))  # (Nodes, NodeLabel+ features + label)
    features_ext = sp.csr_matrix(idx_features_labels_ext[:, 1:], dtype=np.float32)  # (Nodes, features)
    # build features
    idx_crime_side_features_labels = np.genfromtxt("gat_crime_side.txt",
                                                   dtype=np.dtype(str))  # (Nodes, NodeLabel+ features + label)
    crime_side_features = sp.csr_matrix(idx_crime_side_features_labels[:, 1:], dtype=np.float32)  # (Nodes, features)

    # build graph
    num_reg = int(idx_features_labels.shape[0] / 42)
    idx = np.array(idx_features_labels[:num_reg, 0], dtype=np.int32)  # replaced 5
    idx_map = {j: i for i, j in enumerate(idx)}
    edges_unordered = np.genfromtxt("tem_gat_adj.txt", dtype=np.int32)  # changed to tem_gat_adj.txt --> gat_adj.txt
    edges = np.array(list(map(idx_map.get, edges_unordered.flatten())), dtype=np.int32).reshape(edges_unordered.shape)

    adj = sp.coo_matrix((np.ones(edges.shape[0]), (edges[:, 0], edges[:, 1])), shape=(num_reg, num_reg),
                        dtype=np.float32)  # replaced 5
    # build symmetric adjacency matrix
    adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)

    # features = normalize_features(features)
    adj = normalize_adj(adj + sp.eye(adj.shape[0]))

    adj = torch.FloatTensor(np.array(adj.todense()))
    features = torch.FloatTensor(np.array(features.todense()))
    features_ext = torch.FloatTensor(np.array(features_ext.todense()))
    crime_side_features = torch.FloatTensor(np.array(crime_side_features.todense()))

    return adj, features, features_ext, crime_side_features


def normalize_adj(mx):
    """Row-normalize sparse matrix"""
    rowsum = np.array(mx.sum(1))
    r_inv_sqrt = np.power(rowsum, -0.5).flatten()
    r_inv_sqrt[np.isinf(r_inv_sqrt)] = 0.
    r_mat_inv_sqrt = sp.diags(r_inv_sqrt)
    return mx.dot(r_mat_inv_sqrt).transpose().dot(r_mat_inv_sqrt)


def normalize_features(mx):
    """Row-normalize sparse matrix"""
    rowsum = np.array(mx.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    mx = r_mat_inv.dot(mx)
    return mx


def load_data_adj():
    num_labels = 5
    # build features
    idx_features_labels = np.genfromtxt("data/feat.txt", dtype=np.dtype(str))  # (Nodes, NodeLabel+ features + label)
    # features = sp.csr_matrix(idx_features_labels[:, 1:-1], dtype=np.float32)  # (Nodes, features)

    # build graph
    idx = np.array(idx_features_labels[:, 0], dtype=np.int32)
    idx_map = {j: i for i, j in enumerate(idx)}
    edges_unordered = np.genfromtxt("data/adj.txt", dtype=np.int32)  # raw edges
    edges = np.array(list(map(idx_map.get, edges_unordered.flatten())), dtype=np.int32).reshape(
        edges_unordered.shape)  # edges with indexes
    adj = sp.coo_matrix((np.ones(edges.shape[0]), (edges[:, 0], edges[:, 1])), shape=(num_labels, num_labels),
                        dtype=np.float32)
    # build symmetric adjacency matrix
    adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)
    adj = normalize_adj(adj + sp.eye(adj.shape[0]))
    adj = torch.FloatTensor(np.array(adj.todense()))

    return adj


def load_data_regions(target_crime_cat, target_region):
    """
    :param target_crime_cat: starts from 0
    :param target_region: starts from 0
    :return:
    """
    time_step = 120  # consecutive time-step
    add_train = []  # train x's of the regions
    add_test = []  # test x's of the regions
    # com = [6, 23, 27, 31]  # starts from 0
    com = gen_neighbor_index_zero(target_region)
    batch_size = 42
    scaler = MinMaxScaler(feature_range=(-1, 1))
    for i in com:
        loaded_data = torch.from_numpy(np.loadtxt("data/com_crime/r_" + str(i) + ".txt", dtype=np.float)).T
        loaded_data = loaded_data[:, target_crime_cat:target_crime_cat + 1]
        x, y, z, m = create_inout_sequences(loaded_data, time_step)

        x = torch.from_numpy(scaler.fit_transform(x))
        z = torch.from_numpy(scaler.fit_transform(z))
        m = torch.from_numpy(scaler.fit_transform(m))
        y = torch.from_numpy(scaler.fit_transform(y))
        # Divide into train_test data
        train_x_size = int(x.shape[0] * .67)
        train_x = x[: train_x_size, :]  # (batch_size, time-step) = (1386, 120)
        train_y = y[: train_x_size, :]  # (batch_size, time-step) = (1386, 1)
        test_x = x[train_x_size:, :]  # (batch_size, time-step) = (683, 120)
        test_x = test_x[:test_x.shape[0] - 11,
                 :]  # (batch_size, time-step) = (672, 120) -- to make it consistent with the batch size
        test_y = y[train_x_size:, :]  # (batch_size, time-step) = (683, 1)
        test_y = test_y[:test_y.shape[0] - 11, :]

        train_x = train_x.view(int(train_x.shape[0] / batch_size), batch_size, time_step)
        test_x = test_x.view(int(test_x.shape[0] / batch_size), batch_size, time_step)

        train_x = train_x.transpose(2, 1)
        test_x = test_x.transpose(2, 1)

        add_train.append(train_x)
        add_test.append(test_x)

    batch_add_train = []
    batch_add_test = []

    num_batch_train = add_train[0].shape[0]
    len_add_train = len(add_train)
    for i in range(num_batch_train):
        tem = []
        for j in range(len_add_train):
            tem.append(add_train[j][i])
        batch_add_train.append(tem)

    num_batch_test = add_test[0].shape[0]
    len_add_test = len(add_test)
    for i in range(num_batch_test):
        tem = []
        for j in range(len_add_test):
            tem.append(add_test[j][i])
        batch_add_test.append(tem)

    return batch_add_train, batch_add_test


def load_data_regions_external_v2(target_region):
    """
    :param target_region: starts from 0
    :return:
    """
    time_step = 120  # consecutive time-step
    add_train = []  # train x's of the regions
    add_test = []  # test x's of the regions
    # com = [7, 24, 28, 32, 8]  # starts with 1
    com = gen_neighbor_index_one_with_target(target_region)
    batch_size = 42
    poi_data = torch.from_numpy(np.loadtxt("poi.txt", dtype=np.int))
    nfeature = 12
    for i in com:
        loaded_data = torch.from_numpy(np.loadtxt("data/act_ext/taxi" + str(i) + ".txt",
                                                  dtype=np.int)).T  # time step = x; crime type = y # changed from taxi_ to taxi
        loaded_data1 = loaded_data[:, 0:1]
        loaded_data2 = loaded_data[:, 1:2]
        x_in, y_in, z_in, m_in = create_inout_sequences(loaded_data1, time_step)
        x_out, y_out, z_out, m_out = create_inout_sequences(loaded_data2, time_step)

        scale = MinMaxScaler(feature_range=(0, 1))
        # x_in = torch.from_numpy(scale.fit_transform(x_in))
        # y_in = torch.from_numpy(scale.fit_transform(y_in))
        # z_in = torch.from_numpy(scale.fit_transform(z_in))
        # m_in = torch.from_numpy(scale.fit_transform(m_in))

        # x_out = torch.from_numpy(scale.fit_transform(x_out))
        # y_out = torch.from_numpy(scale.fit_transform(y_out))
        # z_out = torch.from_numpy(scale.fit_transform(z_out))
        # m_out = torch.from_numpy(scale.fit_transform(m_out))

        x_in = x_in.unsqueeze(2).double()
        x_out = x_out.unsqueeze(2).double()
        poi = poi_data[i - 1].double()
        # poi_cnt = poi_data[i].sum()
        # poi = poi / poi_cnt
        poi = poi.repeat(x_in.shape[0], time_step, 1)
        x = torch.cat([x_in, x_out, poi], dim=2)

        # Divide into train_test data
        train_x_size = int(x.shape[0] * .67)
        train_x = x[: train_x_size, :, :]  # (batch_size, time-step) = (1386, 120)
        test_x = x[train_x_size:, :, :]  # (batch_size, time-step) = (683, 120)
        test_x = test_x[:test_x.shape[0] - 11, :,
                 :]  # (batch_size, time-step) = (672, 120) -- to make it consistent with the batch size

        train_x = train_x.view(int(train_x.shape[0] / batch_size), batch_size, time_step, nfeature)
        test_x = test_x.view(int(test_x.shape[0] / batch_size), batch_size, time_step, nfeature)

        train_x = train_x.transpose(2, 1)  # (Num, T, B, F)
        test_x = test_x.transpose(2, 1)

        add_train.append(train_x)
        add_test.append(test_x)

    batch_add_train = []
    batch_add_test = []

    num_batch_train = add_train[0].shape[0]
    len_add_train = len(add_train)
    for i in range(num_batch_train):
        tem = []
        for j in range(len_add_train):
            tem.append(add_train[j][i])
        batch_add_train.append(tem)

    num_batch_test = add_test[0].shape[0]
    len_add_test = len(add_test)
    for i in range(num_batch_test):
        tem = []
        for j in range(len_add_test):
            tem.append(add_test[j][i])
        batch_add_test.append(tem)
    # print(len(batch_add_train))
    # print(batch_add_train[0].shape)
    # print(batch_add_train[0][0].shape)
    return batch_add_train, batch_add_test


def load_data_regions_external():
    time_step = 120  # consecutive time-step
    add_train = []  # train x's of the regions
    add_test = []  # test x's of the regions

    com = [6, 23, 27, 31, 7]
    batch_size = 42
    train_batch = 33
    test_batch = 16
    loaded_data = torch.from_numpy(np.loadtxt("poi.txt", dtype=np.int))
    for r in com:
        region = loaded_data[r + 1]
        region = region.repeat(train_batch, time_step, batch_size, 1)
        add_train.append(region)

    for r in com:
        region = loaded_data[r + 1]
        region = region.repeat(test_batch, time_step, batch_size, 1)
        add_test.append(region)

    batch_add_train = []
    batch_add_test = []

    num_batch_train = add_train[0].shape[0]
    len_add_train = len(add_train)
    for i in range(num_batch_train):
        tem = []
        for j in range(len_add_train):
            tem.append(add_train[j][i])
        batch_add_train.append(tem)

    num_batch_test = add_test[0].shape[0]
    len_add_test = len(add_test)
    for i in range(num_batch_test):
        tem = []
        for j in range(len_add_test):
            tem.append(add_test[j][i])
        batch_add_test.append(tem)

    # print(len(batch_add_train))
    # print(batch_add_train[0].shape)
    # print(batch_add_train[0][0].shape)
    return batch_add_train, batch_add_test


def load_data_sides_crime(target_crime_cat, target_region):
    """

    :param target_crime_cat: starts with 0
    :param target_region: starts with 0
    :return:
    """
    time_step = 120  # consecutive time-step
    add_train = []  # train x's of the regions
    add_test = []  # test x's of the regions
    # com = [6, 23, 27, 31, 7]  # starts with 0
    com = gen_neighbor_index_zero_with_target(target_region)
    # side = [2, 3, 3, 4, 4]
    side = gen_com_side_adj_matrix(com)
    batch_size = 42
    scaler = MinMaxScaler(feature_range=(-1, 1))
    for i in range(len(com)):
        loaded_data = torch.from_numpy(np.loadtxt("data/side_crime/s_" + str(side[i]) + ".txt", dtype=np.int)).T
        loaded_data = loaded_data[:, target_crime_cat:target_crime_cat + 1]
        tensor_ones = torch.from_numpy(np.ones((loaded_data.size(0), loaded_data.size(1)), dtype=np.int))
        loaded_data = torch.where(loaded_data > 1, tensor_ones, loaded_data)
        x, y, z, m = create_inout_sequences(loaded_data, time_step)

        x = torch.from_numpy(scaler.fit_transform(x))
        z = torch.from_numpy(scaler.fit_transform(z))
        m = torch.from_numpy(scaler.fit_transform(m))
        y = torch.from_numpy(scaler.fit_transform(y))

        # Divide into train_test data
        train_x_size = int(x.shape[0] * .67)
        train_x = x[: train_x_size, :]  # (batch_size, time-step) = (1386, 120)
        train_y = y[: train_x_size, :]  # (batch_size, time-step) = (1386, 1)
        test_x = x[train_x_size:, :]  # (batch_size, time-step) = (683, 120)
        test_x = test_x[:test_x.shape[0] - 11,
                 :]  # (batch_size, time-step) = (672, 120) -- to make it consistent with the batch size
        test_y = y[train_x_size:, :]  # (batch_size, time-step) = (683, 1)
        test_y = test_y[:test_y.shape[0] - 11, :]

        train_x = train_x.view(int(train_x.shape[0] / batch_size), batch_size, time_step)
        test_x = test_x.view(int(test_x.shape[0] / batch_size), batch_size, time_step)

        train_x = train_x.transpose(2, 1)
        test_x = test_x.transpose(2, 1)

        add_train.append(train_x)
        add_test.append(test_x)

    batch_add_train = []
    batch_add_test = []

    num_batch_train = add_train[0].shape[0]
    len_add_train = len(add_train)
    for i in range(num_batch_train):
        tem = []
        for j in range(len_add_train):
            tem.append(add_train[j][i])
        batch_add_train.append(tem)

    num_batch_test = add_test[0].shape[0]
    len_add_test = len(add_test)
    for i in range(num_batch_test):
        tem = []
        for j in range(len_add_test):
            tem.append(add_test[j][i])
        batch_add_test.append(tem)
    return batch_add_train, batch_add_test


def create_inout_sequences_anomaly(input_data, tw, ncrime):
    forecast = 1
    in_seq1 = torch.from_numpy(np.ones((8000, tw * ncrime), dtype=int))
    out_seq1 = torch.from_numpy(np.ones((8000, forecast * ncrime), dtype=int))

    L = input_data.shape[0]
    for i in range(L - tw - forecast):
        train_seq = input_data[i:i + tw, :]
        in_seq1[i] = train_seq.view(train_seq.shape[0] * train_seq.shape[1])
        # print(train_seq.view(train_seq.shape[0] * train_seq.shape[1]).np())
        train_label = input_data[i + tw:i + tw + forecast, :]
        out_seq1[i] = train_label.view(train_label.shape[0] * train_label.shape[1])
        # in_seq.append(train_seq)
        # out_seq.append(train_label)
    in_seq1 = in_seq1[:i + 1, :]
    out_seq1 = out_seq1[:i + 1, :]

    return in_seq1, out_seq1


def encode_onehot(labels):
    classes = set(labels)
    classes_dict = {c: np.identity(len(classes))[i, :] for i, c in enumerate(classes)}
    labels_onehot = np.array(list(map(classes_dict.get, labels)), dtype=np.int32)
    return labels_onehot


def load_crime_data_regions_MiST2(target_region, target_category):
    """
    :param target_region: starts from 0
    :param target_category: starts from 0
    :return:
    """
    time_step = 120  # consecutive time-step
    # add_train = []  # train x's of the regions
    # add_test = []  # test x's of the regions
    # com = [6, 23, 27, 31, 7]  # starts from 0
    com = gen_neighbor_index_zero_with_target(target_region)
    batch_size = 42
    # for each category
    batch_add_train = []
    batch_add_test = []

    add_train = []  # train x's of the regions
    add_test = []  # test x's of the regions
    for i in range(len(com)):
        train = torch.zeros((8, 33, 42, 120))
        test = torch.zeros((8, 16, 42, 120))
        for j in range(8):
            loaded_data = torch.from_numpy(np.loadtxt("data/com_crime/r_" + str(com[i]) + ".txt", dtype=np.int)).T
            loaded_data = loaded_data[:, j:j + 1]
            tensor_ones = torch.from_numpy(np.ones((loaded_data.size(0), loaded_data.size(1)), dtype=np.int))
            loaded_data = torch.where(loaded_data > 1, tensor_ones, loaded_data)
            x, y, z, m = create_inout_sequences(loaded_data, time_step)

            scale = MinMaxScaler(feature_range=(-1, 1))
            x = torch.from_numpy(scale.fit_transform(x))
            y = torch.from_numpy(scale.fit_transform(y))
            z = torch.from_numpy(scale.fit_transform(z))
            m = torch.from_numpy(scale.fit_transform(m))

            # Divide into train_test data
            train_x_size = int(x.shape[0] * .67)
            train_x = x[: train_x_size, :]  # (batch_size, time-step) = (1386, 120)
            train_y = y[: train_x_size, :]  # (batch_size, time-step) = (1386, 1)
            test_x = x[train_x_size:, :]  # (batch_size, time-step) = (683, 120)
            test_x = test_x[:test_x.shape[0] - 11,
                     :]  # (batch_size, time-step) = (672, 120) -- to make it consistent with the batch size
            test_y = y[train_x_size:, :]  # (batch_size, time-step) = (683, 1)
            test_y = test_y[:test_y.shape[0] - 11, :]

            train_x = train_x.view(int(train_x.shape[0] / batch_size), batch_size, time_step)
            test_x = test_x.view(int(test_x.shape[0] / batch_size), batch_size, time_step)

            # train_x = train_x.transpose(2, 1)
            # test_x = test_x.transpose(2, 1)

            train[j] = train_x
            test[j] = test_x

        train = torch.transpose(train, 1, 0)
        test = torch.transpose(test, 1, 0)
        add_train.append(train)
        add_test.append(test)

    batch_add_train = []
    batch_add_test = []

    num_batch_train = add_train[0].shape[0]  # 33
    len_add_train = len(add_train)  # 5
    for i in range(num_batch_train):
        tem = []
        for j in range(len_add_train):
            tem.append(add_train[j][i])
        batch_add_train.append(tem)
    # print(batch_add_train)
    num_batch_test = add_test[0].shape[0]
    len_add_test = len(add_test)
    for i in range(num_batch_test):
        tem = []
        for j in range(len_add_test):
            tem.append(add_test[j][i])
        batch_add_test.append(tem)
    return batch_add_train, batch_add_test


def gen_com_adj_matrix(target_region):
    adj_matrix = np.zeros((77, 77), dtype=np.int)
    edges_unordered = np.genfromtxt("data/com_adjacency.txt", dtype=np.int32)
    for i in range(edges_unordered.shape[0]):
        src = edges_unordered[i][0] - 1
        dst = edges_unordered[i][1] - 1
        adj_matrix[src][dst] = 1
        adj_matrix[src][dst] = 1
    np.savetxt("data/com_adj_matrix.txt", adj_matrix, fmt="%d")
    return


def gen_com_side_adj_matrix(regions):
    """
    :param regions: a list of regions starting from 0
    :return: sides: a list of sides which are mapped (side, com) starts with 0
    """
    idx = np.loadtxt("data/side_com_adj.txt", dtype=np.int)
    idx_map = {j: i for i, j in iter(idx)}
    side = [idx_map.get(x + 1) % 101 for x in regions]  # As it starts with 0
    return side


def gen_neighbor_index_zero(target_region):
    """
    :param target_region: starts from 0
    :return: indices of neighbors of target region (starts from 0)
    """
    adj_matrix = np.loadtxt("data/com_adj_matrix.txt")
    adj_matrix = adj_matrix[target_region]
    neighbors = []
    for i in range(adj_matrix.shape[0]):
        if adj_matrix[i] == 1:
            neighbors.append(i)
    return neighbors


def gen_neighbor_index_zero_with_target(target_region):
    """
    :param target_region: starts from 0
    :return: indices of neighbors of target region (starts from 0)
    """
    neighbors = gen_neighbor_index_zero(target_region)
    neighbors.append(target_region)
    return neighbors


def gen_neighbor_index_one_with_target(target_region):
    """
    :param target_region: starts from 0
    :return: indices of neighbors of target region (starts from 0)
    """
    neighbors = gen_neighbor_index_zero(target_region)
    neighbors.append(target_region)
    neighbors = [x + 1 for x in neighbors]
    return neighbors


def gen_gat_adj_file(target_region):
    """
    :param target_region: starts from 0
    :return:
    """
    neighbors = gen_neighbor_index_zero(target_region)
    adj_target = torch.zeros(len(neighbors), 2)
    for i in range(len(neighbors)):
        adj_target[i][0] = target_region
        adj_target[i][1] = neighbors[i]
    np.savetxt("tem_gat_adj.txt", adj_target, fmt="%d")
    return


def gen_gat_adj_file_GeoMAN(target_region):
    """
    :param target_region: starts from 0
    :return:
    """
    neighbors = gen_neighbor_index_zero(target_region)
    adj_target = torch.zeros(len(neighbors), 2)
    for i in range(len(neighbors)):
        adj_target[i][0] = target_region
        adj_target[i][1] = neighbors[i]
    np.savetxt("tem_gat_adj_GeoMAN.txt", adj_target, fmt="%d")
    return

def kld(a1, a2):
    # (B, *, A), #(B, *, A)
    a1 = torch.clamp(a1, 0, 1)
    a2 = torch.clamp(a2, 0, 1)
    log_a1 = torch.log(a1 + 1e-10)
    log_a2 = torch.log(a2 + 1e-10)

    kld = a1 * (log_a1 - log_a2)
    kld = kld.sum(-1)

    return kld


def instance_jsd(p, q):
    m = 0.5 * (p + q)
    jsd = 0.5 * (kld(p, m) + kld(q, m))  # for each instance in the batch
    """print("*******************")
    print(p, " ", q)
    print(kld(p, m), " ", kld(q, m))
    print("*******************")"""
    return jsd


def batch_jsd(p, q):
    m = 0.5 * (p + q)
    jsd = 0.5 * (kld(p, m) + kld(q, m))  # for each instance in the batch
    jsd = jsd.unsqueeze(-1)
    # return jsd
    # print(jsd.squeeze(1).mean())
    return jsd.squeeze(1).sum() / 42


def jensen_shannon_distance(p, q):
    """
    method to compute the Jenson-Shannon Distance
    between two probability distributions
    """

    # convert the vectors into numpy arrays in case that they aren't
    p = np.array(p)
    q = np.array(q)

    # calculate m
    m = (p + q) / 2

    # compute Jensen Shannon Divergence
    divergence = (scipy.stats.entropy(p, m, base=np.e) + scipy.stats.entropy(q, m, base=np.e)) / 2

    # compute the Jensen Shannon Distance
    # distance = np.sqrt(divergence)
    return divergence.sum()


def batch_tvd(predictions, targets):  # accepts two Torch tensors... " "
    # print(predictions, targets)
    out = (0.5 * torch.abs(predictions - targets))
    return out
    # return (0.5 * torch.abs(predictions - targets)).sum()


def load_data_regions_GeoMAN(target_crime_cat, target_region, ts):
    """
    :param target_crime_cat: starts from 0
    :param target_region: starts from 0
    :return:
    """
    time_step = ts  # consecutive time-step
    add_train = []  # train x's of the regions
    add_test = []  # test x's of the regions

    # com = [6, 23, 27, 31]  # starts from 0
    com = gen_neighbor_index_zero(target_region)
    com.append(target_region)

    batch_size = 42
    scaler = MinMaxScaler(feature_range=(-1, 1))
    for i in com:
        loaded_data = torch.from_numpy(np.loadtxt("data/com_crime/r_" + str(i) + ".txt", dtype=np.float)).T
        loaded_data = loaded_data[:, target_crime_cat:target_crime_cat + 1]
        x, y = create_inout_sequences_GeoMAN(loaded_data, time_step)

        x = torch.from_numpy(scaler.fit_transform(x))
        y = torch.from_numpy(scaler.fit_transform(y))
        # Divide into train_test data
        train_x_size = int(x.shape[0] * .67)
        sub_train_x = train_x_size - int((train_x_size / batch_size)) * batch_size
        train_x = x[: train_x_size - sub_train_x, :]  # (batch_size, time-step) = (1386, 120)
        train_y = y[: train_x_size - sub_train_x, :]  # (batch_size, time-step) = (1386, 1)

        test_x = x[train_x_size:, :]  # (batch_size, time-step) = (683, 120)
        sub_test_x = test_x.shape[0] - int((test_x.shape[0] / batch_size)) * batch_size
        test_x = test_x[:test_x.shape[0] - sub_test_x,
                 :]  # (batch_size, time-step) = (672, 120) -- to make it consistent with the batch size
        test_y = y[train_x_size:, :]  # (batch_size, time-step) = (683, 1)
        test_y = test_y[:test_y.shape[0] - sub_test_x, :]

        train_x = train_x.view(int(train_x.shape[0] / batch_size), batch_size, time_step)
        test_x = test_x.view(int(test_x.shape[0] / batch_size), batch_size, time_step)

        train_x = train_x.transpose(2, 1)
        test_x = test_x.transpose(2, 1)

        add_train.append(train_x)
        add_test.append(test_x)

    batch_add_train = []
    batch_add_test = []

    num_batch_train = add_train[0].shape[0]
    len_add_train = len(add_train)
    for i in range(num_batch_train):
        tem = []
        for j in range(len_add_train):
            tem.append(add_train[j][i])
        batch_add_train.append(tem)

    num_batch_test = add_test[0].shape[0]
    len_add_test = len(add_test)
    for i in range(num_batch_test):
        tem = []
        for j in range(len_add_test):
            tem.append(add_test[j][i])
        batch_add_test.append(tem)

    return batch_add_train, batch_add_test


def load_data_regions_STGCN(target_crime_cat, target_region):
    """
    :param target_crime_cat: starts from 0
    :param target_region: starts from 0
    :return:
    """
    time_step = 120  # consecutive time-step
    add_train_x = []  # train x's of the regions
    add_train_y = []  # train x's of the regions
    add_test_x = []  # test x's of the regions
    add_test_y = []  # test x's of the regions

    # com = [6, 23, 27, 31]  # starts from 0
    com = gen_neighbor_index_zero(target_region)
    com.append(target_region)

    batch_size = 42
    scaler = MinMaxScaler(feature_range=(-1, 1))
    for i in com:
        loaded_data = torch.from_numpy(np.loadtxt("data/com_crime/r_" + str(i) + ".txt", dtype=np.float)).T
        loaded_data = loaded_data[:, target_crime_cat:target_crime_cat + 1]
        x, y, z, m = create_inout_sequences(loaded_data, time_step)

        x = torch.from_numpy(scaler.fit_transform(x))
        y = torch.from_numpy(scaler.fit_transform(y))

        # Divide into train_test data
        train_x_size = int(x.shape[0] * .67)
        train_x = x[: train_x_size, :]  # (batch_size, time-step) = (1386, 120)
        train_y = y[: train_x_size, :]  # (batch_size, time-step) = (1386, 1)
        test_x = x[train_x_size:, :]  # (batch_size, time-step) = (683, 120)
        test_x = test_x[:test_x.shape[0] - 11,
                 :]  # (batch_size, time-step) = (672, 120) -- to make it consistent with the batch size
        test_y = y[train_x_size:, :]  # (batch_size, time-step) = (683, 1)
        test_y = test_y[:test_y.shape[0] - 11, :]

        # train_x = train_x.transpose(2, 1)
        # test_x = test_x.transpose(2, 1)

        add_train_x.append(train_x)
        add_train_y.append(train_y)
        add_test_x.append(test_x)
        add_test_y.append(test_y)

    train_x = torch.transpose(torch.stack(add_train_x, 0), 1, 0)
    train_x = train_x.view(train_x.shape[0], train_x.shape[1], train_x.shape[2], -1)
    train_y = torch.transpose(torch.stack(add_train_y, 0), 1, 0)
    test_x = torch.transpose(torch.stack(add_test_x, 0), 1, 0)
    test_x = test_x.view(test_x.shape[0], test_x.shape[1], test_x.shape[2], -1)
    test_y = torch.transpose(torch.stack(add_test_y, 0), 1, 0)

    return train_x, train_y, test_x, test_y


def load_data_all_STGCN(target_crime_cat):
    """
    :param target_crime_cat: starts from 0
    :return:
    """
    time_step = 120  # consecutive time-step
    add_train_x = []  # train x's of the regions
    add_train_y = []  # train x's of the regions
    add_test_x = []  # test x's of the regions
    add_test_y = []  # test x's of the regions

    # com = [6, 23, 27, 31]  # starts from 0
    # com = gen_neighbor_index_zero(target_region)
    # com.append(target_region)

    batch_size = 42
    scaler = MinMaxScaler(feature_range=(-1, 1))
    for i in range(77):
        loaded_data = torch.from_numpy(np.loadtxt("data/com_crime/r_" + str(i) + ".txt", dtype=np.float)).T
        loaded_data = loaded_data[:, target_crime_cat:target_crime_cat + 1]
        x, y, z, m = create_inout_sequences(loaded_data, time_step)

        x = torch.from_numpy(scaler.fit_transform(x))
        y = torch.from_numpy(scaler.fit_transform(y))

        # Divide into train_test data
        train_x_size = int(x.shape[0] * .67)
        train_x = x[: train_x_size, :]  # (batch_size, time-step) = (1386, 120)
        train_y = y[: train_x_size, :]  # (batch_size, time-step) = (1386, 1)
        test_x = x[train_x_size:, :]  # (batch_size, time-step) = (683, 120)
        test_x = test_x[:test_x.shape[0] - 11,
                 :]  # (batch_size, time-step) = (672, 120) -- to make it consistent with the batch size
        test_y = y[train_x_size:, :]  # (batch_size, time-step) = (683, 1)
        test_y = test_y[:test_y.shape[0] - 11, :]

        # train_x = train_x.transpose(2, 1)
        # test_x = test_x.transpose(2, 1)

        add_train_x.append(train_x)
        add_train_y.append(train_y)
        add_test_x.append(test_x)
        add_test_y.append(test_y)

    train_x = torch.transpose(torch.stack(add_train_x, 0), 1, 0)
    train_x = train_x.view(train_x.shape[0], train_x.shape[1], train_x.shape[2], -1)
    train_y = torch.transpose(torch.stack(add_train_y, 0), 1, 0)
    test_x = torch.transpose(torch.stack(add_test_x, 0), 1, 0)
    test_x = test_x.view(test_x.shape[0], test_x.shape[1], test_x.shape[2], -1)
    test_y = torch.transpose(torch.stack(add_test_y, 0), 1, 0)
    print(train_x.shape, train_y.shape)
    print(test_x.shape, test_y.shape)
    return train_x, train_y, test_x, test_y


# load_data_all_STGCN(1)


def get_normalized_adj_STGCN(A):
    """
    Returns the degree normalized adjacency matrix.
    """
    A = A + np.diag(np.ones(A.shape[0], dtype=np.float32))
    D = np.array(np.sum(A, axis=1)).reshape((-1,))
    D[D <= 10e-5] = 10e-5  # Prevent infs
    diag = np.reciprocal(np.sqrt(D))
    A_wave = np.multiply(np.multiply(diag.reshape((-1, 1)), A),
                         diag.reshape((1, -1)))
    return A_wave


def gen_adj_matrix_STGCN(target_region):
    """
    Generates the adj_matrix for target region for STGCN
    :param target_region: starts from 0
    :return:
    """
    neighbors = gen_neighbor_index_zero(target_region)
    neighbors.append(target_region)

    adj_target = np.zeros(shape=(len(neighbors), len(neighbors)), dtype=np.int)
    for i in range(len(neighbors)):
        adj_target[i][len(neighbors) - 1] = 1
        adj_target[i][i] = 1
        adj_target[len(neighbors) - 1][i] = 1
    return adj_target


def load_data_regions_ratio(target_crime_cat, target_region, ratio, sub_train, sub_test):
    """
    loads data of neighbor regions for experimenting on train test ratio
    :param sub_test:
    :param sub_train:
    :param ratio:
    :param target_crime_cat: starts from 0
    :param target_region: starts from 0
    :return:
    """
    time_step = 120  # consecutive time-step
    add_train = []  # train x's of the regions
    add_test = []  # test x's of the regions
    # com = [6, 23, 27, 31]  # starts from 0
    com = gen_neighbor_index_zero(target_region)
    batch_size = 42
    scaler = MinMaxScaler(feature_range=(-1, 1))
    for i in com:
        loaded_data = torch.from_numpy(np.loadtxt("data/com_crime/r_" + str(i) + ".txt", dtype=np.float)).T
        loaded_data = loaded_data[:, target_crime_cat:target_crime_cat + 1]
        x, y, z, m = create_inout_sequences(loaded_data, time_step)

        x = torch.from_numpy(scaler.fit_transform(x))
        z = torch.from_numpy(scaler.fit_transform(z))
        m = torch.from_numpy(scaler.fit_transform(m))
        y = torch.from_numpy(scaler.fit_transform(y))

        # Divide into train_test data
        train_x_size = int(x.shape[0] * ratio)

        train_x = x[: train_x_size, :]  # (batch_size, time-step) = (1386, 120)
        train_y = y[: train_x_size, :]  # (batch_size, time-step) = (1386, 1)
        test_x = x[train_x_size:, :]  # (batch_size, time-step) = (683, 120)
        test_y = y[train_x_size:, :]  # (batch_size, time-step) = (683, 1)

        train_x = x[: train_x.shape[0] - sub_train, :]  # (batch_size, time-step) = (1386, 120)
        train_y = y[: train_x.shape[0] - sub_train, :]  # (batch_size, time-step) = (1386, 1)
        test_x = test_x[:test_x.shape[0] - sub_test, :]  # (batch_size, time-step) = (672, 120)
        test_y = test_y[:test_y.shape[0] - sub_test, :]

        train_x = train_x.view(int(train_x.shape[0] / batch_size), batch_size, time_step)
        test_x = test_x.view(int(test_x.shape[0] / batch_size), batch_size, time_step)

        train_x = train_x.transpose(2, 1)
        test_x = test_x.transpose(2, 1)

        add_train.append(train_x)
        add_test.append(test_x)

    batch_add_train = []
    batch_add_test = []

    num_batch_train = add_train[0].shape[0]
    len_add_train = len(add_train)
    for i in range(num_batch_train):
        tem = []
        for j in range(len_add_train):
            tem.append(add_train[j][i])
        batch_add_train.append(tem)

    num_batch_test = add_test[0].shape[0]
    len_add_test = len(add_test)
    for i in range(num_batch_test):
        tem = []
        for j in range(len_add_test):
            tem.append(add_test[j][i])
        batch_add_test.append(tem)

    return batch_add_train, batch_add_test


def load_data_regions_external_v2_ratio(target_region, ratio, sub_train, sub_test):
    """
    :param sub_test:
    :param sub_train:
    :param ratio:
    :param target_region: starts from 0
    :return:
    """
    time_step = 120  # consecutive time-step
    add_train = []  # train x's of the regions
    add_test = []  # test x's of the regions
    # com = [7, 24, 28, 32, 8]  # starts with 1
    com = gen_neighbor_index_one_with_target(target_region)
    batch_size = 42
    poi_data = torch.from_numpy(np.loadtxt("poi.txt", dtype=np.int))
    nfeature = 12
    for i in com:
        loaded_data = torch.from_numpy(np.loadtxt("data/act_ext/taxi" + str(i) + ".txt",
                                                  dtype=np.int)).T  # time step = x; crime type = y # changed from taxi_ to taxi
        loaded_data1 = loaded_data[:, 0:1]
        loaded_data2 = loaded_data[:, 1:2]
        x_in, y_in, z_in, m_in = create_inout_sequences(loaded_data1, time_step)
        x_out, y_out, z_out, m_out = create_inout_sequences(loaded_data2, time_step)

        scale = MinMaxScaler(feature_range=(0, 1))
        # x_in = torch.from_numpy(scale.fit_transform(x_in))
        # y_in = torch.from_numpy(scale.fit_transform(y_in))
        # z_in = torch.from_numpy(scale.fit_transform(z_in))
        # m_in = torch.from_numpy(scale.fit_transform(m_in))

        # x_out = torch.from_numpy(scale.fit_transform(x_out))
        # y_out = torch.from_numpy(scale.fit_transform(y_out))
        # z_out = torch.from_numpy(scale.fit_transform(z_out))
        # m_out = torch.from_numpy(scale.fit_transform(m_out))

        x_in = x_in.unsqueeze(2).double()
        x_out = x_out.unsqueeze(2).double()
        poi = poi_data[i - 1].double()
        # poi_cnt = poi_data[i].sum()
        # poi = poi / poi_cnt
        poi = poi.repeat(x_in.shape[0], time_step, 1)
        x = torch.cat([x_in, x_out, poi], dim=2)

        # Divide into train_test data
        train_x_size = int(x.shape[0] * ratio)

        train_x = x[: train_x_size, :, :]  # (batch_size, time-step) = (1386, 120)
        test_x = x[train_x_size:, :, :]  # (batch_size, time-step) = (683, 120)

        train_x = train_x[:train_x.shape[0] - sub_train, :, :]
        test_x = test_x[:test_x.shape[0] - sub_test, :,
                 :]  # (batch_size, time-step) = (672, 120) -- to make it consistent with the batch size

        train_x = train_x.view(int(train_x.shape[0] / batch_size), batch_size, time_step, nfeature)
        test_x = test_x.view(int(test_x.shape[0] / batch_size), batch_size, time_step, nfeature)

        train_x = train_x.transpose(2, 1)  # (Num, T, B, F)
        test_x = test_x.transpose(2, 1)

        add_train.append(train_x)
        add_test.append(test_x)

    batch_add_train = []
    batch_add_test = []

    num_batch_train = add_train[0].shape[0]
    len_add_train = len(add_train)
    for i in range(num_batch_train):
        tem = []
        for j in range(len_add_train):
            tem.append(add_train[j][i])
        batch_add_train.append(tem)

    num_batch_test = add_test[0].shape[0]
    len_add_test = len(add_test)
    for i in range(num_batch_test):
        tem = []
        for j in range(len_add_test):
            tem.append(add_test[j][i])
        batch_add_test.append(tem)
    # print(len(batch_add_train))
    # print(batch_add_train[0].shape)
    # print(batch_add_train[0][0].shape)
    return batch_add_train, batch_add_test


def load_data_sides_crime_ratio(target_crime_cat, target_region, ratio, sub_train, sub_test):
    """

    :param sub_test:
    :param sub_train:
    :param ratio:
    :param target_crime_cat: starts with 0
    :param target_region: starts with 0
    :return:
    """
    time_step = 120  # consecutive time-step
    add_train = []  # train x's of the regions
    add_test = []  # test x's of the regions
    # com = [6, 23, 27, 31, 7]  # starts with 0
    com = gen_neighbor_index_zero_with_target(target_region)
    # side = [2, 3, 3, 4, 4]
    side = gen_com_side_adj_matrix(com)
    batch_size = 42
    scaler = MinMaxScaler(feature_range=(-1, 1))
    for i in range(len(com)):
        loaded_data = torch.from_numpy(np.loadtxt("data/side_crime/s_" + str(side[i]) + ".txt", dtype=np.int)).T
        loaded_data = loaded_data[:, target_crime_cat:target_crime_cat + 1]
        tensor_ones = torch.from_numpy(np.ones((loaded_data.size(0), loaded_data.size(1)), dtype=np.int))
        loaded_data = torch.where(loaded_data > 1, tensor_ones, loaded_data)
        x, y, z, m = create_inout_sequences(loaded_data, time_step)

        x = torch.from_numpy(scaler.fit_transform(x))
        z = torch.from_numpy(scaler.fit_transform(z))
        m = torch.from_numpy(scaler.fit_transform(m))
        y = torch.from_numpy(scaler.fit_transform(y))

        # Divide into train_test data
        train_x_size = int(x.shape[0] * ratio)
        train_x = x[: train_x_size, :]  # (batch_size, time-step) = (1386, 120)
        train_y = y[: train_x_size, :]  # (batch_size, time-step) = (1386, 1)
        test_x = x[train_x_size:, :]  # (batch_size, time-step) = (683, 120)
        test_y = y[train_x_size:, :]  # (batch_size, time-step) = (683, 1)

        train_x = train_x[:train_x.shape[0] - sub_train,
                  :]  # (batch_size, time-step) = (672, 120) -- to make it consistent with the batch size
        train_y = train_y[:train_y.shape[0] - sub_train, :]
        test_x = test_x[:test_x.shape[0] - sub_test,
                 :]  # (batch_size, time-step) = (672, 120) -- to make it consistent with the batch size
        test_y = test_y[:test_y.shape[0] - sub_test, :]

        train_x = train_x.view(int(train_x.shape[0] / batch_size), batch_size, time_step)
        test_x = test_x.view(int(test_x.shape[0] / batch_size), batch_size, time_step)

        train_x = train_x.transpose(2, 1)
        test_x = test_x.transpose(2, 1)

        add_train.append(train_x)
        add_test.append(test_x)

    batch_add_train = []
    batch_add_test = []

    num_batch_train = add_train[0].shape[0]
    len_add_train = len(add_train)
    for i in range(num_batch_train):
        tem = []
        for j in range(len_add_train):
            tem.append(add_train[j][i])
        batch_add_train.append(tem)

    num_batch_test = add_test[0].shape[0]
    len_add_test = len(add_test)
    for i in range(num_batch_test):
        tem = []
        for j in range(len_add_test):
            tem.append(add_test[j][i])
        batch_add_test.append(tem)
    return batch_add_train, batch_add_test


def load_data_regions_MVGCN(target_crime_cat, target_region):
    """
    :param target_crime_cat: starts from 0
    :param target_region: starts from 0
    :return:
    """
    time_step = 120  # consecutive time-step
    add_train_x = []  # train x's of the regions
    add_train_z = []  # train x's of the regions
    add_train_m = []  # train x's of the regions
    add_train_y = []  # train x's of the regions

    add_test_x = []  # test x's of the regions
    add_test_z = []  # train x's of the regions
    add_test_m = []  # train x's of the regions
    add_test_y = []  # train x's of the regions

    # com = [6, 23, 27, 31]  # starts from 0
    com = gen_neighbor_index_zero(target_region)
    com.append(target_region)

    batch_size = 42
    scaler = MinMaxScaler(feature_range=(-1, 1))
    for i in com:
        loaded_data = torch.from_numpy(np.loadtxt("data/com_crime/r_" + str(i) + ".txt", dtype=np.float)).T
        loaded_data = loaded_data[:, target_crime_cat:target_crime_cat + 1]
        x, y, z, m = create_inout_sequences(loaded_data, time_step)

        x = torch.from_numpy(scaler.fit_transform(x))
        z = torch.from_numpy(scaler.fit_transform(z))
        m = torch.from_numpy(scaler.fit_transform(m))
        y = torch.from_numpy(scaler.fit_transform(y))

        # Divide into train_test data
        train_x_size = int(x.shape[0] * .67)
        train_x = x[: train_x_size, :]  # (batch_size, time-step) = (1386, 120)
        train_z = z[: train_x_size, :]  # (batch_size, time-step) = (1386, 120)
        train_m = m[: train_x_size, :]
        train_y = y[: train_x_size, :]  # (batch_size, time-step) = (1386, 1)

        test_x = x[train_x_size:, :]  # (batch_size, time-step) = (683, 120)
        test_x = test_x[:test_x.shape[0] - 11,
                 :]  # (batch_size, time-step) = (672, 120) -- to make it consistent with the batch size

        test_z = z[train_x_size:, :]  # (batch_size, time-step) = (683, 120)
        test_z = test_z[:test_z.shape[0] - 11,
                 :]  # (batch_size, time-step) = (672, 120) -- to make it consistent with the batch size

        test_m = m[train_x_size:, :]  # (batch_size, time-step) = (683, 120)
        test_m = test_m[:test_m.shape[0] - 11,
                 :]  # (batch_size, time-step) = (672, 120) -- to make it consistent with the batch size

        test_y = y[train_x_size:, :]  # (batch_size, time-step) = (683, 1)
        test_y = test_y[:test_y.shape[0] - 11, :]

        train_x = train_x.view(int(train_x.shape[0] / batch_size), batch_size, time_step)
        train_z = train_z.view(int(train_z.shape[0] / batch_size), batch_size, train_z.shape[1])
        train_m = train_m.view(int(train_m.shape[0] / batch_size), batch_size, train_m.shape[1])
        train_y = train_y.view(int(train_y.shape[0] / batch_size), batch_size, train_y.shape[1])

        test_x = test_x.view(int(test_x.shape[0] / batch_size), batch_size, time_step)
        test_z = test_z.view(int(test_z.shape[0] / batch_size), batch_size, test_z.shape[1])
        test_m = test_m.view(int(test_m.shape[0] / batch_size), batch_size, test_m.shape[1])
        test_y = test_y.view(int(test_y.shape[0] / batch_size), batch_size, test_y.shape[1])

        """train_x = train_x.transpose(2, 1)
        train_z = train_z.transpose(2, 1)
        train_m = train_m.transpose(2, 1)
        train_y = train_y.transpose(2, 1)
        test_x = test_x.transpose(2, 1)
        test_y = test_y.transpose(2, 1)"""

        add_train_x.append(train_x)
        add_train_z.append(train_z)
        add_train_m.append(train_m)
        add_train_y.append(train_y)

        add_test_x.append(test_x)
        add_test_z.append(test_z)
        add_test_m.append(test_m)
        add_test_y.append(test_y)

    """print(add_train_x[0].shape)
    print(add_train_z[0].shape)
    print(add_train_m[0].shape)
    print(add_train_y[0].shape)
    print(add_test_x[0].shape)
    print(add_test_y[0].shape)"""

    """batch_add_train_x = []
    batch_add_train_z = []
    batch_add_train_m = []
    batch_add_train_y = []
    batch_add_test_x = []
    batch_add_test_y = []

    num_batch_train = add_train_x[0].shape[0]
    len_add_train = len(add_train_x)
    for i in range(num_batch_train):
        tem = []
        for j in range(len_add_train):
            tem.append(add_train_x[j][i])
        batch_add_train_x.append(tem)

    num_batch_test = add_test_x[0].shape[0]
    len_add_test = len(add_test_y)
    for i in range(num_batch_test):
        tem = []
        for j in range(len_add_test):
            tem.append(add_test_x[j][i])
        batch_add_test_x.append(tem)"""

    add_train_x = torch.stack(add_train_x, 2)
    add_train_z = torch.stack(add_train_z, 2)
    add_train_m = torch.stack(add_train_m, 2)
    add_train_y = torch.stack(add_train_y, 2)
    add_test_x = torch.stack(add_test_x, 2)
    add_test_z = torch.stack(add_test_z, 2)
    add_test_m = torch.stack(add_test_m, 2)
    add_test_y = torch.stack(add_test_y, 2)

    return add_train_x, add_train_z, add_train_m, add_train_y, add_test_x, add_test_z, add_test_m, add_test_y


def load_data_external_MVGCN(target_region):
    """
    :param target_region: starts from 0
    :return:
    """
    time_step = 120  # consecutive time-step
    add_train = []  # train x's of the regions
    add_test = []  # test x's of the regions
    # com = [7, 24, 28, 32, 8]  # starts with 1
    com = gen_neighbor_index_one_with_target(target_region)
    batch_size = 42
    poi_data = torch.from_numpy(np.loadtxt("poi.txt", dtype=np.int))
    nfeature = 12

    loaded_data = torch.from_numpy(np.loadtxt("data/act_ext/taxi" + str(target_region+1) + ".txt",
                                              dtype=np.int)).T  # time step = x; crime type = y # changed from taxi_ to taxi
    loaded_data1 = loaded_data[:, 0:1]
    loaded_data2 = loaded_data[:, 1:2]
    x_in, y_in, z_in, m_in = create_inout_sequences(loaded_data1, time_step)
    x_out, y_out, z_out, m_out = create_inout_sequences(loaded_data2, time_step)

    scale = MinMaxScaler(feature_range=(0, 1))
    # x_in = torch.from_numpy(scale.fit_transform(x_in))
    # y_in = torch.from_numpy(scale.fit_transform(y_in))
    # z_in = torch.from_numpy(scale.fit_transform(z_in))
    # m_in = torch.from_numpy(scale.fit_transform(m_in))

    # x_out = torch.from_numpy(scale.fit_transform(x_out))
    # y_out = torch.from_numpy(scale.fit_transform(y_out))
    # z_out = torch.from_numpy(scale.fit_transform(z_out))
    # m_out = torch.from_numpy(scale.fit_transform(m_out))

    x_in = x_in.unsqueeze(2).double()
    x_out = x_out.unsqueeze(2).double()
    poi = poi_data[target_region - 1].double()
    # poi_cnt = poi_data[i].sum()
    # poi = poi / poi_cnt
    poi = poi.repeat(x_in.shape[0], time_step, 1)
    x = torch.cat([x_in, x_out, poi], dim=2)

    # Divide into train_test data
    train_x_size = int(x.shape[0] * .67)
    train_x = x[: train_x_size, :, :]  # (batch_size, time-step) = (1386, 120)
    test_x = x[train_x_size:, :, :]  # (batch_size, time-step) = (683, 120)
    test_x = test_x[:test_x.shape[0] - 11, :,
             :]  # (batch_size, time-step) = (672, 120) -- to make it consistent with the batch size

    train_x = train_x.view(int(train_x.shape[0] / batch_size), batch_size, time_step, nfeature)
    test_x = test_x.view(int(test_x.shape[0] / batch_size), batch_size, time_step, nfeature)

    return train_x, test_x

