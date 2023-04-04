from __future__ import absolute_import

import torch.nn as nn
from models.sem_graph_conv import SemGraphConv
from models.graph_non_local import GraphNonLocal

from functools import reduce
class _GraphConv(nn.Module):
    def __init__(self, adj ,input_dim, output_dim, p_dropout=None):
        super(_GraphConv, self).__init__()

        self.gconv = SemGraphConv(input_dim, output_dim, adj)
        self.bn = nn.BatchNorm1d(output_dim)
        self.relu = nn.ReLU()

        if p_dropout is not None:
            self.dropout = nn.Dropout(p_dropout)
        else:
            self.dropout = None

    def forward(self, x):
        x = self.gconv(x).transpose(1, 2)
        x = self.bn(x).transpose(1, 2)
        if self.dropout is not None:
            x = self.dropout(self.relu(x))

        x = self.relu(x)
        return x
class _ChannelTrans(nn.Module):
    def __init__(self, input_dim, output_dim,):
        super(_ChannelTrans, self).__init__()

        self.conv = nn.Conv1d(input_dim,output_dim,1,1,0)

    def forward(self,x):
        x = self.conv(x)
        return x



class _ResGraphConv(nn.Module):
    def __init__(self, adj, input_dim, output_dim, hid_dim, p_dropout):
        super(_ResGraphConv, self).__init__()

        self.gconv1 = _GraphConv(adj, input_dim, hid_dim, p_dropout)
        self.gconv2 = _GraphConv(adj, hid_dim, output_dim, p_dropout)

    def forward(self, x):
        residual = x
        out = self.gconv1(x)
        out = self.gconv2(out)
        return residual + out


class _GraphNonLocal(nn.Module):
    def __init__(self, hid_dim, grouped_order, restored_order, group_size):
        super(_GraphNonLocal, self).__init__()

        self.nonlocal1 = GraphNonLocal(hid_dim)
        # self.grouped_order = grouped_order
        # self.restored_order = restored_order

    def forward(self, x):
        # out = x[:, self.grouped_order, :]
        out = self.nonlocal1(x.transpose(1, 2)).transpose(1, 2)
        # out = out[:, self.restored_order, :]
        return out


class HGN(nn.Module):
    def __init__(self, adj,adj_48,adj_96, hid_dim, coords_dim=(2, 3), num_layers=4, nodes_group=None, p_dropout=None):
        super(HGN, self).__init__()

 
        grouped_order = None
        restored_order = None
        group_size =1
      

        _gconv_input = [_GraphConv(adj, coords_dim[0], hid_dim, p_dropout=p_dropout)]
        _gconv_1_1_layers = []
        _gconv_1_2_layers = []
        _gconv_1_3_layers = []
        _gconv_1_4_layers = []
        _gconv_2_1_layers = []
        _gconv_2_2_layers = []
        _gconv_2_3_layers = []
        _gconv_2_4_layers = []
        _gconv_3_3_layers = []
        _gconv_3_4_layers = []


        self.fc_1_2_1 = _ChannelTrans(17,48)
        _gconv_1_1_layers.append(_ResGraphConv(adj, hid_dim, hid_dim, hid_dim, p_dropout=p_dropout))
        _gconv_1_1_layers.append(_GraphNonLocal(hid_dim, grouped_order, restored_order, group_size))

        self.fc_1_2_2 = _ChannelTrans(17,48)

        _gconv_1_2_layers.append(_ResGraphConv(adj, hid_dim, hid_dim, hid_dim, p_dropout=p_dropout))
        _gconv_1_2_layers.append(_GraphNonLocal(hid_dim, grouped_order, restored_order, group_size))

        self.fc_1_2_3 = _ChannelTrans(17,48)
        self.fc_1_3_3 = _ChannelTrans(17,96)

        _gconv_1_3_layers.append(_ResGraphConv(adj, hid_dim, hid_dim, hid_dim, p_dropout=p_dropout))
        _gconv_1_3_layers.append(_GraphNonLocal(hid_dim, grouped_order, restored_order, group_size))

        self.fc_1_2_4 = _ChannelTrans(17,48)
        self.fc_1_3_4 = _ChannelTrans(17,96)

        _gconv_1_4_layers.append(_ResGraphConv(adj, hid_dim, hid_dim, hid_dim, p_dropout=p_dropout))
        _gconv_1_4_layers.append(_GraphNonLocal(hid_dim, grouped_order, restored_order, group_size))

        self.fc_1_3_5 = _ChannelTrans(17,96)
        self.fc_1_2_5 = _ChannelTrans(17,48)
 

        _gconv_2_1_layers.append(_ResGraphConv(adj_48, hid_dim, hid_dim, hid_dim, p_dropout=p_dropout))
        _gconv_2_1_layers.append(_GraphNonLocal(hid_dim, grouped_order, restored_order, group_size))

        self.fc_2_1_2 = _ChannelTrans(48,17)

        _gconv_2_2_layers.append(_ResGraphConv(adj_48, hid_dim, hid_dim, hid_dim, p_dropout=p_dropout))
        _gconv_2_2_layers.append(_GraphNonLocal(hid_dim, grouped_order, restored_order, group_size))

        self.fc_2_1_3 = _ChannelTrans(48,17)
        self.fc_2_3_3 = _ChannelTrans(48,96)

        _gconv_2_3_layers.append(_ResGraphConv(adj_48, hid_dim, hid_dim, hid_dim, p_dropout=p_dropout))
        _gconv_2_3_layers.append(_GraphNonLocal(hid_dim, grouped_order, restored_order, group_size))

        self.fc_2_1_4 = _ChannelTrans(48,17)
        
        self.fc_2_3_4 = _ChannelTrans(48,96)

        _gconv_2_4_layers.append(_ResGraphConv(adj_48, hid_dim, hid_dim, hid_dim, p_dropout=p_dropout))
        _gconv_2_4_layers.append(_GraphNonLocal(hid_dim, grouped_order, restored_order, group_size))
        self.fc_2_1_5 = _ChannelTrans(48,17)
        self.fc_2_3_5 = _ChannelTrans(48,96)

        _gconv_3_3_layers.append(_ResGraphConv(adj_96, hid_dim, hid_dim, hid_dim, p_dropout=p_dropout))
        _gconv_3_3_layers.append(_GraphNonLocal(hid_dim, grouped_order, restored_order, group_size))

        self.fc_3_1_4 =  _ChannelTrans(96,17)
        self.fc_3_2_4 =  _ChannelTrans(96,48)

        
        _gconv_3_4_layers.append(_ResGraphConv(adj_96, hid_dim, hid_dim, hid_dim, p_dropout=p_dropout))
        _gconv_3_4_layers.append(_GraphNonLocal(hid_dim, grouped_order, restored_order, group_size))
        self.fc_3_1_5 =  _ChannelTrans(96,17)
        self.fc_3_2_5 =  _ChannelTrans(96,48)


        self.gconv_input = nn.Sequential(*_gconv_input)
        self.gconv_layers_1_1 = nn.Sequential(*_gconv_1_1_layers)
        self.gconv_layers_1_2 = nn.Sequential(*_gconv_1_2_layers)
        self.gconv_layers_1_3 = nn.Sequential(*_gconv_1_3_layers)
        self.gconv_layers_1_4 = nn.Sequential(*_gconv_1_4_layers)


        self.gconv_layers_2_1 = nn.Sequential(*_gconv_2_1_layers)
        self.gconv_layers_2_2 = nn.Sequential(*_gconv_2_2_layers)
        self.gconv_layers_2_3 = nn.Sequential(*_gconv_2_3_layers)
        self.gconv_layers_2_4 = nn.Sequential(*_gconv_2_4_layers)

        self.gconv_layers_3_3 = nn.Sequential(*_gconv_3_3_layers)
        self.gconv_layers_3_4 = nn.Sequential(*_gconv_3_4_layers)

        self.gconv_output = SemGraphConv(hid_dim, coords_dim[1], adj)
        self.gconv_output_48 = SemGraphConv(hid_dim, coords_dim[1], adj_48)
        self.gconv_output_96 = SemGraphConv(hid_dim, coords_dim[1], adj_96)

    def forward(self, x):
        f_1_0 = self.gconv_input(x)
        f_2_0 = self.fc_1_2_1(f_1_0)
        # layers1 
        f_1_1_p = self.gconv_layers_1_1(f_1_0)
        f_2_1_p = self.gconv_layers_2_1(f_2_0)

        f_1_1_f = f_1_1_p
        f_2_1_f = f_2_1_p

        # layers2 
        f_1_1_f += self.fc_2_1_2(f_2_1_p)
        f_2_1_f += self.fc_1_2_2(f_1_1_p)

        f_1_2_p = self.gconv_layers_1_2(f_1_1_f)
        f_2_2_p = self.gconv_layers_2_2(f_2_1_f)


        #layers3
        f_1_2_f = f_1_2_p
        f_2_2_f = f_2_2_p
        f_1_2_f += self.fc_2_1_3(f_2_2_p)
        f_2_2_f += self.fc_1_2_3(f_1_2_p)
        f_3_2_f = self.fc_1_3_3(f_1_2_p) + self.fc_2_3_3(f_2_2_p)

        f_1_3_p = self.gconv_layers_1_3(f_1_2_f)
        f_2_3_p = self.gconv_layers_2_3(f_2_2_f)
        f_3_3_p = self.gconv_layers_3_3(f_3_2_f)

        #layers4
        f_1_3_f = f_1_3_p
        f_2_3_f = f_2_3_p
        f_3_3_f = f_3_3_p

        f_1_3_f += self.fc_2_1_4(f_2_3_p) +self.fc_3_1_4(f_3_3_p)
        f_2_3_f += self.fc_1_2_4(f_1_3_p) +self.fc_3_2_4(f_3_3_p)
        f_3_3_f += self.fc_1_3_4(f_1_3_p) + self.fc_2_3_4(f_2_3_p)

        f_1_4_p = self.gconv_layers_1_4(f_1_3_f)
        f_2_4_p = self.gconv_layers_2_4(f_2_3_f)
        f_3_4_p = self.gconv_layers_3_4(f_3_3_f)

        f_1_4_p += self.fc_2_1_5(f_2_4_p) +self.fc_3_1_5(f_3_4_p)
        f_2_4_p += self.fc_1_2_5(f_1_4_p) +self.fc_3_2_5(f_3_4_p)
        f_3_4_p += self.fc_1_3_5(f_1_4_p) +self.fc_2_3_5(f_2_4_p)

        out = self.gconv_output(f_1_4_p)
        out_48 = self.gconv_output_48(f_2_4_p)
        out_96 = self.gconv_output_96(f_3_4_p)
        return out,out_48,out_96
