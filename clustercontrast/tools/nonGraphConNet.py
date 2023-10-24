# -*- coding: UTF-8 -*-
'''
@Project ：FuseDSI
@File ：nonGraphConNet.py
@Author ：棋子
@Date ：2023/8/28 20:37
@Software: PyCharm
'''
"""
Thanks very much for Graph Convolution Based Efficient Re-Ranking for Visual Retrieval
Graph Convolution Based Efficient Re-Ranking for Visual Retrieval: https://arxiv.org/pdf/2306.08792.pdf.
"""

import numpy as np
import torch
from torch import nn

class GCN(nn.Module):
    def __init__(self, k1=5, k2=3):
        super(GCN, self).__init__()
        self.k1 = k1
        self.k2 = k2
        self.beta1 = self.beta2 = 0.2
        self.scale = 0.3

    def forward(self, X, labels_cam):
        """Run GCR for one iteration."""
        unique_labels_cam = np.unique(labels_cam)
        index_dic = {item: [] for item in unique_labels_cam}
        for labels_index, i in enumerate(labels_cam):
            index_dic[i].append(labels_index)

        # compute global feat
        sim = X.mm(X.t())

        if self.scale != 1.0:
            rank = torch.argsort(-sim, axis=1)
            S = torch.zeros(sim.shape).cuda()
            for i in range(0, X.shape[0]):
                S[i, rank[i, :self.k1]] = torch.exp(sim[i, rank[i, :self.k1]] / self.beta1)
            D_row = torch.sqrt(1. / torch.sum(S, axis=1))
            D_col = torch.sqrt(1. / torch.sum(S, axis=0))
            L = torch.outer(D_row, D_col) * S
            global_X = L.mm(X)
        else:
            global_X = 0.0

        if self.scale != 0.0:
            # compute cross camera feat
            for i in range(0, X.shape[0]):
                tmp = sim[i, i]
                sim[i, index_dic[labels_cam[i]]] = -2
                sim[i, i] = tmp

            rank = torch.argsort(-sim, axis=1)
            S = torch.zeros(sim.shape).cuda()
            for i in range(0, X.shape[0]):
                S[i, rank[i, :self.k2]] = torch.exp(sim[i, rank[i, :self.k2]] / self.beta2)
                S[i, i] = torch.exp(sim[i, i] / self.beta2)
            D_row = torch.sqrt(1. / torch.sum(S, axis=1))
            D_col = torch.sqrt(1. / torch.sum(S, axis=0))
            L = torch.outer(D_row, D_col) * S
            cross_X = L.mm(X)
        else:
            cross_X = 0.0

        X = self.scale * cross_X + (1 - self.scale) * global_X
        X /= torch.linalg.norm(X, ord=2, axis=1, keepdims=True)

        return X


