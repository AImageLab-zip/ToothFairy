#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Author: Yue Wang
@Contact: yuewangx@mit.edu
@File: model.py
@Time: 2018/10/13 6:35 PM

Modified by 
@Author: An Tao, Ziyi Wu
@Contact: ta19@mails.tsinghua.edu.cn, dazitu616@gmail.com
@Time: 2022/7/30 7:49 PM
"""


import os
import sys
import copy
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
import torch.distributions as dist



def knn(x, k):
    inner = -2*torch.matmul(x.transpose(2, 1), x)
    xx = torch.sum(x**2, dim=1, keepdim=True)
    pairwise_distance = -xx - inner - xx.transpose(2, 1)

    idx = pairwise_distance.topk(k=k, dim=-1)[1]   # (batch_size, num_points, k)

    return idx

def k_similar(x, k):
    x_norm = torch.norm(x, p=2, dim=1, keepdim=True)
    x = x / x_norm
    inner = torch.matmul(x.transpose(2, 1), x)
    idx = inner.topk(k=k, dim=-1)[1]   # (batch_size, num_points, k)
    return idx

def get_graph_feature(x, k=20, idx=None, dim9=False, return_idx=False):
    # x: (B, dim, np)
    batch_size = x.size(0)
    num_points = x.size(2)
    x = x.view(batch_size, -1, num_points)
    if idx is None:
        if dim9 == False:
            idx = knn(x, k=k)   # (batch_size, num_points, k)
        else:
            idx = knn(x[:, :3], k=k)
        # else:
        #     x_in = x[:, 6:].clone()
        #     x_in[:, :2] *= 10
        #     idx = knn(x_in, k=k)
    index = idx
    device = x.device

    idx_base = torch.arange(0, batch_size, device=device).view(-1, 1, 1)*num_points
    idx = idx + idx_base   # idx: (B, np, k)
    idx = idx.view(-1)  # idx: B*np*k
    _, num_dims, _ = x.size()

    x = x.transpose(2, 1).contiguous()   # (batch_size, num_points, num_dims)  -> (batch_size*num_points, num_dims) #   batch_size * num_points * k + range(0, batch_size*num_points)
    feature = x.view(batch_size*num_points, -1)[idx, :]  # (B*np*k, dim)
    feature = feature.view(batch_size, num_points, k, num_dims) # (B, np, k, dim)
    x = x.view(batch_size, num_points, 1, num_dims).repeat(1, 1, k, 1)  # (B, np, k, dim)
    feature = torch.cat((feature-x, x), dim=3).permute(0, 3, 1, 2).contiguous()

    if return_idx:
        return feature, index
    else:
        return feature      # (batch_size, 2*num_dims, num_points, k)


def index_points(points, idx):
    """

    Input:
        points: input points data, [B, N, C]
        idx: sample index data, [B, S]
    Return:
        new_points:, indexed points data, [B, S, C]
    """
    device = points.device
    B = points.shape[0]
    view_shape = list(idx.shape)
    view_shape[1:] = [1] * (len(view_shape) - 1)
    repeat_shape = list(idx.shape)
    repeat_shape[0] = 1
    batch_indices = torch.arange(B, dtype=torch.long).to(device).view(view_shape).repeat(repeat_shape)
    new_points = points[batch_indices, idx, :]
    return new_points

class GraphAttention(nn.Module):
    def __init__(self,feature_dim,out_dim, K):
        super(GraphAttention, self).__init__()
        self.dropout = 0.6
        self.conv = nn.Sequential(nn.Conv2d(feature_dim * 2, out_dim, kernel_size=1, bias=False),
                                     nn.BatchNorm2d(out_dim),
                                     nn.LeakyReLU(negative_slope=0.2))
        self.K=K

    def forward(self, Graph_index, x, feature):

        B, C, N = x.shape
        x = x.contiguous().view(B, N, C)
        feature = feature.permute(0,2,3,1)
        neighbor_feature = index_points(x, Graph_index)
        centre = x.view(B, N, 1, C).expand(B, N, self.K, C)
        delta_f = torch.cat([centre-neighbor_feature, neighbor_feature], dim=3).permute(0,3,2,1)
        e = self.conv(delta_f)
        e = e.permute(0,3,2,1)
        attention = F.softmax(e, dim=2) # [B, npoint, nsample,D]
        graph_feature = torch.sum(torch.mul(attention, feature),dim = 2) .permute(0,2,1)
        return graph_feature

def cal_same(A, B):
    # A: (bs, np, k)
    # B: (bs, np, R)
    A_expanded = A.unsqueeze(-1)
    B_expanded = B.unsqueeze(-2)

    matches = (A_expanded == B_expanded).any(-1)
    result = matches.sum(dim=-1)

    return result


def select_indices_from_uniform_distribution(B, N, M, k, device):
    # Step 1: 生成0到M之间的离散均匀分布
    probs = torch.ones(B, N, M, device=device) / M
    selected_indices = torch.multinomial(probs.view(B * N, M), k, replacement=False)

    # Step 3: 重塑为 (B, N, k) 的形状
    selected_indices = selected_indices.view(B, N, k)

    selected_indices = selected_indices.sort(dim=-1)[0]

    return selected_indices

def select_indices_from_gaussian_distribution(mean, std, M, k):
    
    B, N = mean.shape

    # Step 1: 生成0到M之间的离散正态分布
    x_values = torch.arange(0, M, dtype=torch.float32, device=mean.device) / M  # 归一化的0~M
    dist_normal = dist.Normal(mean.unsqueeze(-1), std.unsqueeze(-1))  # 每个位置的均值和标准差

    # Step 2: 为每个索引计算选中的概率P(x/M)，假设P(x) = N(mean, std**2)的值
    probs = torch.exp(dist_normal.log_prob(x_values))  # P(x/M)
    
    # Step 3: 归一化概率
    probs = probs / probs.sum(dim=-1, keepdim=True)

    # Step 4: 选择k个索引
    # 使用multinomial对整个(B, N, M)的概率分布进行抽样

    selected_indices = torch.multinomial(probs.view(B * N, M), k, replacement=False)

    # Step 5: 重塑为 (B, N, k) 的形状
    selected_indices = selected_indices.view(B, N, k)

    selected_indices = selected_indices.sort(dim=-1)[0]

    selected_probs = torch.gather(probs, dim=2, index=selected_indices)

    return selected_indices, probs

class SA_Layer(nn.Module):
    def __init__(self, channels):
        super(SA_Layer, self).__init__()
        self.q_conv = nn.Conv1d(channels, channels // 4, 1, bias=False)
        self.k_conv = nn.Conv1d(channels, channels // 4, 1, bias=False)
        self.q_conv.weight = self.k_conv.weight 
        self.v_conv = nn.Conv1d(channels, channels, 1)
        self.trans_conv = nn.Conv1d(channels, channels, 1)
        self.after_norm = nn.BatchNorm1d(channels)
        self.act = nn.ReLU()
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        x_q = self.q_conv(x).permute(0, 2, 1) # b, n, c 
        x_k = self.k_conv(x)# b, c, n        
        x_v = self.v_conv(x)
        energy = torch.bmm(x_q, x_k) # b, n, n 
        attention = self.softmax(energy)
        attention = attention / (1e-9 + attention.sum(dim=1, keepdims=True))
        x_r = torch.bmm(x_v, attention) # b, c, n 
        x_r = self.act(self.after_norm(self.trans_conv(x - x_r)))
        x = x + x_r
        return x


class GaussianBlock(nn.Module):
    def __init__(self, in_channels):
        super(GaussianBlock, self).__init__()
        # self.SA = SA_Layer(in_channels)
        self.conv_mean = nn.Sequential(
            nn.Conv1d(in_channels, in_channels*2, kernel_size=1, bias=False),
            nn.InstanceNorm1d(in_channels*2),
            nn.Conv1d(in_channels*2, 1, kernel_size=1, bias=False),
            nn.Sigmoid()
        )
        self.conv_std = nn.Sequential(
            nn.Conv1d(in_channels, in_channels*2, kernel_size=1, bias=False),
            nn.InstanceNorm1d(in_channels*2),
            nn.Conv1d(in_channels*2, 1, kernel_size=1, bias=False),
            nn.Softplus()
        )

    def forward(self, x):
        # x = self.SA(x)

        mean = self.conv_mean(x)
        std = self.conv_std(x) + 0.1
        # std = F.softplus(std) + 0.1
        return mean, std



class DeformableBlock(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, pos_dim, k, reception, mode="gaussian"):
        super(DeformableBlock, self).__init__()
        self.k = k
        self.reception = reception
        self.conv_s = nn.Sequential(
            nn.Conv2d(in_channels*2, hidden_channels, kernel_size=1, bias=False),
            nn.InstanceNorm2d(hidden_channels),
            nn.LeakyReLU(negative_slope=0.2))
        self.conv_p = nn.Sequential(
            nn.Conv2d(in_channels*2+pos_dim*2, hidden_channels, kernel_size=1, bias=False),
            nn.InstanceNorm2d(hidden_channels),
            nn.LeakyReLU(negative_slope=0.2))
        self.conv = nn.Sequential(
            nn.Conv1d(hidden_channels*2, out_channels, kernel_size=1, bias=False),
            nn.InstanceNorm1d(out_channels),
            nn.LeakyReLU(negative_slope=0.2))
        self.gaussianblock = GaussianBlock(hidden_channels)
        self.mode = mode

        
    def forward(self, x_s, x, x0, widx):
        x_s = get_graph_feature(x_s, k=self.k, return_idx=False)
        x_s = self.conv_s(x_s)
        x_s = x_s.max(dim=-1, keepdim=False)[0]
        cur_idx = widx[:, :, :self.reception]

        mean, std = self.gaussianblock(x_s)
 
        if self.mode == "gaussian":
            random_indices, probs = select_indices_from_gaussian_distribution(mean.squeeze(1), std.squeeze(1), self.reception, self.k)
            selected_widx = cur_idx.gather(2, random_indices)
            random_probs = probs.gather(2, random_indices)
            random_probs = random_probs.unsqueeze(1)
            
            x = torch.cat((x, x0), dim=1)
            x_p = get_graph_feature(x, k=self.k, idx=selected_widx)
            x_p = self.conv_p(x_p)
            # x_p = x_p.max(dim=-1, keepdim=False)[0]
            x_p = (x_p * random_probs).sum(dim=-1) / random_probs.sum(dim=-1)
        elif self.mode == "uniform":
            x = torch.cat((x, x0), dim=1)
            random_indices = select_indices_from_uniform_distribution(x.size(0), x.size(2), self.reception, self.k, x.device)
            selected_widx = cur_idx.gather(2, random_indices)
            x_p = get_graph_feature(x, k=self.k, idx=selected_widx)
            x_p = self.conv_p(x_p)
            x_p = x_p.max(dim=-1, keepdim=False)[0]

        x = self.conv(torch.cat((x_s, x_p), dim=1))
        return x_s, x, torch.cat((mean, std), dim=1)


class DeformableTeethNet(nn.Module):
    def __init__(self, num_classes, k=32, emb_dims=1024, dropout=0.5, pos_dim=3):
        super(DeformableTeethNet, self).__init__()
        
        self.k = k
        self.receptions = [k, k*4, k*16]

        self.conv1 = nn.Sequential(
            nn.Conv2d(12, 64, kernel_size=1, bias=False),
            nn.InstanceNorm2d(64),
            nn.LeakyReLU(negative_slope=0.2), 
            nn.Conv2d(64, 64, kernel_size=1, bias=False),
            nn.InstanceNorm2d(64),
            nn.LeakyReLU(negative_slope=0.2))
        
        self.deformableconv1 = DeformableBlock(64, 64, 64, pos_dim, self.k, self.receptions[1])
        self.deformableconv2 = DeformableBlock(64, 64, 64, pos_dim, self.k, self.receptions[2])

        self.conv2 = nn.Sequential(
            nn.Conv1d(64*3, emb_dims, kernel_size=1, bias=False),
            nn.InstanceNorm1d(emb_dims),
            nn.LeakyReLU(negative_slope=0.2))
        self.conv3 = nn.Sequential(
            nn.Conv1d(emb_dims+64*3+pos_dim, 512, kernel_size=1, bias=False),
            nn.InstanceNorm1d(512),
            nn.LeakyReLU(negative_slope=0.2))
        self.conv4 = nn.Sequential(
            nn.Conv1d(512, 256, kernel_size=1, bias=False),
            nn.InstanceNorm1d(256),
            nn.LeakyReLU(negative_slope=0.2))
        self.dp1 = nn.Dropout(p=dropout)
        self.conv5 = nn.Conv1d(256+pos_dim, num_classes, kernel_size=1, bias=False)
    
    def forward(self, x):
        bs = x.size(0)
        npoint = x.size(2)
        device = x.device
        x0 = x[:, :3, :] # (bs, 3, npoint)
        widx = knn(x0, k=self.receptions[-1])
        # x0 = self.pos_encoder(x0) # (bs, 1, npoint)
        # emb_x0 = x0.unsqueeze(3).repeat(1, 1, 1, self.k)

        x = get_graph_feature(x, k=self.k, idx=widx[:, :, :self.k])
        x = self.conv1(x)
        x1 = x.max(dim=-1, keepdim=False)[0]

        x_s, x2, feature1 = self.deformableconv1(x1, x1, x0, widx)
        x_s, x3, feature2 = self.deformableconv2(x_s, x2, x0, widx)

        x = torch.cat((x1, x2, x3), dim=1)
        x = self.conv2(x)
        x = x.max(dim=-1, keepdim=True)[0]

        x = x.repeat(1, 1, npoint)
        x = torch.cat((x, x1, x2, x3, x0), dim=1)
        x = self.conv3(x)

        # x = torch.cat((x), dim=1)
        x = self.conv4(x)
        x = self.dp1(x)

        x = torch.cat((x, x0), dim=1)
        x = self.conv5(x)
        x = x.transpose(2, 1).contiguous()
        
        # prob = torch.max(prob1, prob2)

        return x, feature1, feature2

