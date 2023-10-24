# -*- coding: UTF-8 -*-
'''
@Project ：FuseDSI
@File ：PairwiseSemanticInformation.py
@Author ：棋子
@Date ：2023/10/24 14:50
@Software: PyCharm
'''
"""
Thanks very much for 'Embedding Transfer with Label Relaxation for Improved Metric Learning' contribution
Embedding Transfer with Label Relaxation for Improved Metric Learning: https://arxiv.org/pdf/2103.14908.pdf.
"""
import numpy as np
import torch
import torch.nn as nn

class PSI(nn.Module):
    def __init__(self, sigma=1, delta=1, topk=10):
        super(PSI, self).__init__()
        self.sigma = sigma
        self.delta = delta
        self.topk = topk

    def k_reciprocal_neigh(self, initial_rank, i, topk):
        forward_k_neigh_index = initial_rank[i, :topk]
        backward_k_neigh_index = initial_rank[forward_k_neigh_index, :topk]
        fi = np.where(backward_k_neigh_index == i)[0]
        return forward_k_neigh_index[fi]

    def forward(self, s_emb, t_emb):
        N = len(s_emb)
        S_dist = torch.cdist(s_emb, s_emb)
        S_dist = S_dist / S_dist.mean(1, keepdim=True)

        with torch.no_grad():
            T_dist = torch.cdist(t_emb, t_emb)
            W = torch.exp(-T_dist.pow(2) / self.sigma)

            identity_matrix = torch.eye(N).cuda(non_blocking=True)
            pos_weight = (W) * (1 - identity_matrix)
            neg_weight = (1 - W) * (1 - identity_matrix)

        pull_losses = torch.relu(S_dist).pow(2) * pos_weight
        push_losses = torch.relu(self.delta - S_dist).pow(2) * neg_weight

        loss = (pull_losses.sum() + push_losses.sum()) / (len(s_emb) * (len(s_emb) - 1))

        return loss



