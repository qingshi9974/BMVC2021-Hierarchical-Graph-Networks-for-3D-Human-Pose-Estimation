# Copyright (c) Liuhao Ge. All Rights Reserved.
r"""
Graph utilities
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import os.path as osp
import numpy as np
import scipy.sparse as sp
import torch
from coarsening import coarsen, laplacian, perm_index_reverse, lmax_L, rescale_L


def normalize_sparse_mx(mx):
    """Row-normalize sparse matrix"""
    rowsum = np.array(mx.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    mx = r_mat_inv.dot(mx)
    return mx


def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(np.vstack((sparse_mx.row,
                                          sparse_mx.col))).long()
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)


def build_graph(hand_tri, num_vertex):
    """
    :param hand_tri: T x 3
    :return: adj: sparse matrix, V x V (torch.sparse.FloatTensor)
    """
    num_tri = hand_tri.shape[0]
    edges = np.empty((num_tri * 3, 2))
    for i_tri in range(num_tri):
        edges[i_tri * 3] = hand_tri[i_tri, :2]
        edges[i_tri * 3 + 1] = hand_tri[i_tri, 1:]
        edges[i_tri * 3 + 2] = hand_tri[i_tri, [0, 2]]

    adj = sp.coo_matrix((np.ones(edges.shape[0]), (edges[:, 0], edges[:, 1])),
                        shape=(num_vertex, num_vertex), dtype=np.float32)

    adj = adj - (adj > 1) * 1.0

    # build symmetric adjacency matrix
    adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)

    # adj = normalize_sparse_mx(adj + sp.eye(adj.shape[0]))
    # adj = sparse_mx_to_torch_sparse_tensor(adj)

    return adj
    
def adj_mx_from_edges(num_pts, edges, sparse=True):
    edges = np.array(edges, dtype=np.int32)
    data, i, j = np.ones(edges.shape[0]), edges[:, 0], edges[:, 1]
    adj_mx = sp.coo_matrix((data, (i, j)), shape=(num_pts, num_pts), dtype=np.float32)

    # build symmetric adjacency matrix
    adj_mx = adj_mx + adj_mx.T.multiply(adj_mx.T > adj_mx) - adj_mx.multiply(adj_mx.T > adj_mx)
    adj_mx = normalize_sparse_mx(adj_mx + sp.eye(adj_mx.shape[0]))
    if sparse:
        adj_mx = sparse_mx_to_torch_sparse_tensor(adj_mx)
    else:
        adj_mx = torch.tensor(adj_mx.todense(), dtype=torch.float)
    return adj_mx

def build_adj(joint_num, skeleton, flip_pairs):
    adj_matrix = np.zeros((joint_num, joint_num))
    for line in skeleton:
        adj_matrix[line] = 1
        adj_matrix[line[1], line[0]] = 1
    # for lr in flip_pairs:
    #     adj_matrix[lr] = 1
    #     adj_matrix[lr[1], lr[0]] = 1
    # adj_matrix =sp.csr_matrix(adj_matrix)

    # adj_matrix = adj_matrix+ adj_matrix.T.multiply(adj_matrix.T > adj_matrix) - adj_matrix.multiply(adj_matrix.T >adj_matrix)
    # adj_matrix = normalize_sparse_mx(adj_matrix + sp.eye(adj_matrix.shape[0]))

    return adj_matrix 


def build_coarse_graphs(mesh_face, joint_num, skeleton, flip_pairs, levels=9,is_coarse =1):
    joint_adj = build_adj(joint_num, skeleton, flip_pairs)
    # Build graph
    mesh_adj = build_graph(mesh_face, mesh_face.max() + 1)
   
    graph_Adj, graph_L, graph_perm,parents,graphs= coarsen(mesh_adj, levels=levels)
    # import pdb
    # pdb.set_trace()
    #print(graph_perm[0].shape)
    # print(joint_adj)
    input_Adj = sp.csr_matrix(joint_adj)
    input_Adj.eliminate_zeros()
    # import pdb
    # pdb.set_trace()
    input_L = laplacian(input_Adj, normalized=True)

    # input_Adj = sp.csr_matrix(joint_adj)
    # input_Adj = input_Adj+ input_Adj.T.multiply(input_Adj.T > input_Adj) - input_Adj.multiply(input_Adj.T >input_Adj)
    input_Adj = normalize_sparse_mx(input_Adj + sp.eye(input_Adj.shape[0]))
    input_Adj = torch.tensor(input_Adj.todense(), dtype=torch.float)
    # print(input_Adj)
    # import pdb
    # pdb.set_trace()


    graph_L[-1] = input_L
    graph_Adj[-1] = input_Adj

    adj_48 = sp.csr_matrix((graph_Adj[-2]>0)*1.0)
    # adj_48 = adj_48.eliminate_zeros()
    adj_48 = normalize_sparse_mx(adj_48)
    adj_48 = torch.tensor(adj_48.todense(), dtype=torch.float)
    graph_Adj[-2] =adj_48 

    adj_96 = sp.csr_matrix((graph_Adj[-3]>0)*1.0)
    # adj_96 = adj_96.eliminate_zeros()
    adj_96 = normalize_sparse_mx(adj_96)
    adj_96 = torch.tensor(adj_96.todense(), dtype=torch.float)
    graph_Adj[-3]=adj_96

    # Compute max eigenvalue of graph Laplacians, rescale Laplacian
    graph_lmax = []
    renewed_lmax = []
    for i in range(levels):
        graph_lmax.append(lmax_L(graph_L[i]))
        graph_L[i] = rescale_L(graph_L[i], graph_lmax[i])
    #     renewed_lmax.append(lmax_L(graph_L[i]))

    if is_coarse :
        graph_mask = torch.from_numpy((np.array(graph_perm[7]) <83).astype(float)).float()
        graph_mask = graph_mask.unsqueeze(-1).expand(-1, 3)  # V 
        return graph_Adj, graph_L, graph_perm, perm_index_reverse(graph_perm[7]),perm_index_reverse(graph_perm[8]),parents,graphs,graph_mask
        sk
    else:
        graph_mask = torch.from_numpy((np.array(graph_perm[0]) <mesh_face.max() + 1).astype(float)).float()
        graph_mask = graph_mask.unsqueeze(-1).expand(-1, 3)  # V 
        return graph_Adj, graph_L, graph_perm, perm_index_reverse(graph_perm[0]),parents,graphs,graph_mask


def sparse_python_to_torch(sp_python):
    L = sp_python.tocoo()
    indices = np.column_stack((L.row, L.col)).T
    indices = indices.astype(np.int64)
    indices = torch.from_numpy(indices)
    indices = indices.type(torch.LongTensor)
    L_data = L.data.astype(np.float32)
    L_data = torch.from_numpy(L_data)
    L_data = L_data.type(torch.FloatTensor)
    L = torch.sparse.FloatTensor(indices, L_data, torch.Size(L.shape))

    return L


class my_sparse_mm(torch.autograd.Function):
    """
    this function is forked from https://github.com/xbresson/spectral_graph_convnets
    Implementation of a new autograd function for sparse variables,
    called "my_sparse_mm", by subclassing torch.autograd.Function
    and implementing the forward and backward passes.
    """

    def forward(self, W, x):  # W is SPARSE
        print("CHECK sparse W: ", W.is_cuda)
        print("CHECK sparse x: ", x.is_cuda)
        self.save_for_backward(W, x)
        y = torch.mm(W, x)
        return y

    def backward(self, grad_output):
        W, x = self.saved_tensors
        grad_input = grad_output.clone()
        grad_input_dL_dW = torch.mm(grad_input, x.t())
        grad_input_dL_dx = torch.mm(W.t(), grad_input)
        return grad_input_dL_dW, grad_input_dL_dx
