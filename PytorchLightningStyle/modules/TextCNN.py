#!/usr/bin/python3
# -*- coding: utf-8 -*-
# @Time    : 2023/2/19 16:43
# @Author  : lxy15058247683@aliyun.com
# @FileName: TextCNN.py
# @Copyright: MIT

import os
from typing import Dict
import numpy as np
import torch
import torch.nn as nn
import torchmetrics
from torch.utils.data import DataLoader
from tqdm import tqdm, trange
from tensorboardX import SummaryWriter
from transformers import BertTokenizer
import PytorchLightningStyle.PreProcess.DataPreProcess as DPP


class TextCNN(nn.Module):
    def __init__(self, num_embeddings: int, embedding_dim: int, in_channels: int,
                 out_channels: int, kernels: Dict, num_classes: int, *args, **kwargs):
        super(TextCNN, self).__init__(*args, **kwargs)
        self.embedding_dim = embedding_dim
        # 定义用到的组件
        self.embedding = nn.Embedding(num_embeddings, embedding_dim)
        self.convs = nn.ModuleList([nn.Conv2d(in_channels=in_channels, out_channels=out_channels,
                                kernel_size=(kernel["kernel_size"], 1),
                                padding=(kernel["padding"], 0),
                                stride=(kernel["stride"], 1)) for kernel in kernels])
        self.maxpooling = nn.MaxPool1d(kernel_size=(kernels[-1]["kernel_size"], embedding_dim),
                                       stride=(kernels[-1]["stride"], 1))
        self.classifier = nn.Sequential(
                nn.Linear(in_features=embedding_dim, out_features=num_classes),
                nn.Softmax(dim=-1)
            )

    # 定义正向传播，即计算图
    def forward(self, features):
        ids = features["tokens"]['input_ids'].unsqueeze(1)
        bsz = ids.shape[0]
        feature_map = self.embedding(ids)
        conv1_out = self.convs[0](feature_map)
        conv2_out = self.convs[1](feature_map)
        conv3_out = self.convs[2](feature_map)
        concat = torch.concat([conv1_out, conv2_out, conv3_out], dim=1)
        concat = concat.view(bsz, -1, self.embedding_dim).sum(1)
        return concat


