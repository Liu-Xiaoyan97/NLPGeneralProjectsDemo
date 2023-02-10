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
import PreProcess.DataPreProcess as DPP


class TextCNN(nn.Module):
    def __init__(self, num_embeddings: int, embedding_dim: int, max_seq_len: int, in_channels: int,
                 out_channels: int, kernels: Dict, num_classes:int, lr: float,
                 weight_decay: float, *args, **kwargs):
        super(TextCNN, self).__init__(*args, **kwargs)
        self.embedding_dim = embedding_dim
        # 定义用到的组件
        self.embedding = nn.Embedding(num_embeddings, embedding_dim)
        self.convs = [nn.Conv2d(in_channels=in_channels, out_channels=out_channels,
                                kernel_size=(kernel["kernel_size"], 1),
                                padding=(kernel["padding"], 0),
                                stride=(kernel["stride"], 1)) for kernel in kernels]
        self.maxpooling = nn.MaxPool1d(kernel_size=(kernels[-1]["kernel_size"], embedding_dim),
                                       stride=(kernels[-1]["stride"], 1))
        self.classifier = nn.Sequential(
                nn.Linear(in_features=embedding_dim, out_features=num_classes),
                nn.Softmax(dim=-1)
            )
        # 定义损失函数
        self.loss = nn.CrossEntropyLoss()
        # 定义优化器
        self.optimizer = torch.optim.Adam(self.parameters(), lr=lr, weight_decay=weight_decay)
        # 定义评价指标 torchmetrics里还有很多评价指标可供选择
        self.metrices = torchmetrics.Accuracy()

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
        logits = self.classifier(concat)
        return logits

    # 训练版本号获取
    def dir_path_process(self, logdir: str):
        if os.path.exists(logdir+self.__class__.__name__) is False:
            os.mkdir(logdir+self.__class__.__name__)
        log_dir = os.listdir(logdir+self.__class__.__name__)
        if log_dir == []:
            version = 0
        else:
            version = [x.strip("version_") for x in log_dir]
            version = int(version[-1])+1
        log_dir = logdir+self.__class__.__name__+"/version_"+str(version)
        return log_dir

    # 训练函数
    def trainepoch(self, logdir: str, data_path: str, epoch: int = 2, batch_size: int = 3, shuffle: bool = True):
        log_dir = self.dir_path_process(logdir)
        writer = SummaryWriter(log_dir)
        train_set = DPP.OwnDataset(data_path)
        trainloader = DataLoader(train_set, batch_size=batch_size, shuffle=shuffle)
        train_metrics_epoch = torch.tensor(np.NAN)
        for i in range(epoch):
            with trange(trainloader.__len__()+1) as t:
                for step, features in enumerate(trainloader):
                    logits = self.forward(features)
                    loss = self.loss(logits, features["labels"])
                    logits_indices = torch.argmax(logits, dim=-1)
                    train_metrics_batch = self.metrices(logits_indices, features["labels"])
                    loss.backward()
                    self.optimizer.step()
                    t.set_description("epoch  %00d"%i)
                    t.set_postfix_str(["step_loss %.3f"%(loss.item()),
                                       "step_metric %.3f"%(train_metrics_batch.item()),
                                       "epoch_metric %.3f"%(train_metrics_epoch.item())])
                    writer.add_scalar("step_loss", loss)
                    t.update()
                train_metrics_epoch = self.metrices.compute()
                writer.add_scalar("epoch_loss", train_metrics_epoch)
                self.metrices.reset()
                torch.save(self.state_dict(), log_dir+"/model.pth")
        return True

    # 测试函数
    def test(self, logdir: str, data_path: str, model_dir, batch_size: int = 10):
        log_dir = self.dir_path_process(logdir)
        writer = SummaryWriter(log_dir)
        if data_path is None:
            self.load_state_dict(torch.load(model_dir))
        test_set = DPP.OwnDataset(data_path)
        testloader = DataLoader(test_set, batch_size=batch_size)
        test_metrics_epoch = torch.tensor(np.NAN)
        with trange(testloader.__len__()+1) as t:
            for step, features in enumerate(testloader):
                logits = self.forward(features)
                logits_indices = torch.argmax(logits, dim=-1)
                test_metrics_batch = self.metrices(logits_indices, features["labels"])
                t.set_description("test")
                t.set_postfix_str(["step_metric %.3f"%(test_metrics_batch.item()),
                                       "epoch_metric %.3f"%(test_metrics_epoch.item())])
                t.update()
            test_metrics_epoch = self.metrices.compute()
            writer.add_scalar("test_metrics", test_metrics_epoch)
            self.metrices.reset()

    # 预测函数
    def predict(self, text: str, model_dir: str):
        tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
        tokenizer.add_special_tokens({'pad_token': '[PAD]'})
        features = {"tokens":
            tokenizer.__call__(text=text, padding=True, truncation=True, max_length=128)}
        features["tokens"]["input_ids"] = torch.Tensor(features["tokens"]["input_ids"]).unsqueeze(0).long()
        self.load_state_dict(torch.load(model_dir))
        logits = self.forward(features)
        logits_indices = torch.argmax(logits, dim=-1)
        return {
                "text": text,
                "label": logits_indices
        }


