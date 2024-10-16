# -*- coding: UTF-8 -*-
"""
@Project : Test_Sim 
@File    : signal_diffusion.py
@Author  : Nan Chen
@Date    : 2024-01-15 20:29 
"""

import torch
from torch.linalg import matrix_exp


def compute_laplacian_torch(adjacency_matrix, normalized=True):
    if not torch.is_tensor(adjacency_matrix):
        raise TypeError("Input must be a PyTorch tensor")
    n = adjacency_matrix.shape[0]
    identity_matrix = torch.eye(n, device=adjacency_matrix.device)
    degree_matrix = torch.diag(adjacency_matrix.sum(dim=1))
    if normalized:
        inverse_sqrt_degree_matrix = torch.pow(degree_matrix, -0.5)
        inverse_sqrt_degree_matrix[torch.isinf(inverse_sqrt_degree_matrix)] = 0
        laplacian_matrix = identity_matrix - torch.mm(torch.mm(inverse_sqrt_degree_matrix, adjacency_matrix),
                                                      inverse_sqrt_degree_matrix)
    else:
        laplacian_matrix = degree_matrix - adjacency_matrix

    return laplacian_matrix

def signal_diffuse_torch(adj, t=0.1, num=2, device="cuda:1"):
    lap_matrix = compute_laplacian_torch(adj)
    m = lap_matrix.size(0)
    signal_matrix = torch.zeros((m, m), device=device)

    for i in range(m):

        current_ = torch.zeros(m, device=device)
        current_[i] = 1

        result = torch.matmul(matrix_exp(-t * lap_matrix), current_)
        _, sorted_indices = torch.sort(result, descending=True)
        sorted_heat = torch.zeros_like(result)
        sorted_heat[sorted_indices[:num]] = 1

        signal_matrix[i] = sorted_heat
    return signal_matrix
