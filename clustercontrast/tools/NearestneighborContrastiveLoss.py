# -*- coding: UTF-8 -*-
'''
@Project ：FuseDSI
@File ：NearestneighborContrastiveLoss.py
@Author ：棋子
@Date ：2023/10/24 14:34
@Software: PyCharm
'''
"""
Thanks very much for Supervised Contrastive Learning contribution
Supervised Contrastive Learning: https://arxiv.org/pdf/2004.11362.pdf.
"""
import torch
import torch.nn as nn

def find_k_labels(feats, k=8):
    sim = feats.mm(feats.t())
    values, indices = torch.topk(sim, k=k)
    mask = torch.zeros(256, 256).cuda()
    mask.scatter_(1, indices, 1)
    return mask

class NNConLoss(nn.Module):
    def __init__(self, temperature=0.07, base_temperature=0.07, k=5):
        super(NNConLoss, self).__init__()
        self.temperature = temperature
        self.base_temperature = base_temperature
        self.k = k

    def forward(self, features, feat_t_g):
        device = (torch.device('cuda')
                  if features.is_cuda
                  else torch.device('cpu'))

        mask = find_k_labels(feat_t_g, self.k).to(device)

        if len(features.shape) == 2:
            features = features.unsqueeze(dim=1)
        else:
            raise ValueError('`features` needs to be [B, D]')

        batch_size = features.shape[0]

        contrast_count = features.shape[1]
        contrast_feature = torch.cat(torch.unbind(features, dim=1), dim=0)

        anchor_feature = contrast_feature
        anchor_count = contrast_count

        # compute logits
        anchor_dot_contrast = torch.div(torch.matmul(anchor_feature, contrast_feature.T), self.temperature)

        # for numerical stability
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()

        mask = mask.repeat(anchor_count, contrast_count)
        logits_mask = torch.scatter(
            torch.ones_like(mask),
            1,
            torch.arange(batch_size * anchor_count).view(-1, 1).to(device),
            0
        )
        mask = mask * logits_mask
        # compute log_prob
        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))

        # compute mean of log-likelihood over positive
        mean_log_prob_pos = (mask * log_prob).sum(1) / torch.where(mask.sum(1) == 0, torch.ones_like(mask.sum(1)), mask.sum(1))

        # loss
        loss = - (self.temperature / self.base_temperature) * mean_log_prob_pos
        loss = loss.view(anchor_count, batch_size).mean()

        return loss
