#!/usr/bin/python3
# -*- coding: utf-8 -*-
# @Time    : 2023/2/19 16:47
# @Author  : lxy15058247683@aliyun.com
# @FileName: main.py
# @Copyright: MIT
from typing import Dict, Any, List
import argparse
from omegaconf import OmegaConf
import torch
import torchmetrics
from pytorch_lightning import LightningModule
from pytorch_lightning.utilities.types import STEP_OUTPUT
from torch import nn
import pytorch_lightning as pl

from PytorchLightningStyle.modules.ProjectionLayer import ProjectionLayer
from PytorchLightningStyle.modules.Former import Former
from PytorchLightningStyle.PreProcess import DataModules


class ModulePLStyle(LightningModule):
    def __init__(self,  num_classes: int,
                 embedding_dim: int = 420, lr: float = 1e-3, weight_decay: float = 3e-3,
                 max_len: int = 64, hidden_dim: int = 768, num_heads: int = 10,
                 n_layers: int = 2, dropout_rate: float = 0.4, *args, **kwargs):
        super(ModulePLStyle, self).__init__(*args, **kwargs)
        self.model = nn.Sequential(
            ProjectionLayer(max_len, embedding_dim, hidden_dim),
            # Encoder(EncoderLayer(d_model, c(attn), c(ff), dropout_rate), n_layers),
            Former(embedding_dim, hidden_dim, num_heads, n_layers, dropout_rate)
        )

        self.classifier = nn.Sequential(
            nn.Linear(in_features=embedding_dim, out_features=num_classes),
            # 10 * 64 * 2 得到的好像是一个10*64*2？
            nn.Softmax(dim=-1)
        )
        self.lr = lr
        self.weight_decay = weight_decay
        self.loss = nn.CrossEntropyLoss()
        self.metrices = torchmetrics.Accuracy(task="multiclass", num_classes=num_classes)

    def training_step(self, batch: dict, batch_idx: int) -> STEP_OUTPUT:
        logits = self.classifier(self.model(batch))
        loss = self.loss(logits, batch["labels"].long().view(-1))
        logits_indices = torch.argmax(logits, dim=-1)
        train_metrics_batch = self.metrices(logits_indices, batch["labels"].view(-1))
        self.log("train loss", loss, prog_bar=True, on_step=True, on_epoch=True)
        self.log("train metrics", train_metrics_batch, prog_bar=True, on_step=False, on_epoch=True)
        self.metrices.reset()

    def test_step(self, batch: dict, batch_idx: int) -> STEP_OUTPUT:
        logits = self.classifier(self.model(batch))
        logits_indices = torch.argmax(logits, dim=-1)
        # 原本为 test_metrics_batch = self.metrices(logits_indices, batch["labels"].long().view(-1))
        test_metrics_batch = self.metrices(logits_indices, batch["labels"].view(-1))

        self.log("test metrics", test_metrics_batch, prog_bar=True, on_step=True, on_epoch=True)
        self.metrices.reset()

    def validation_step(self, batch: dict, batch_idx: int) -> STEP_OUTPUT:
        logits = self.classifier(self.model(batch))
        loss = self.loss(logits, batch["labels"].long().view(-1))
        logits_indices = torch.argmax(logits, dim=-1)
        # 原来是这个样子的val_metrics_batch = self.metrices(logits_indices.view(-1), batch["labels"].view(-1))
        val_metrics_batch = self.metrices(logits_indices, batch["labels"].view(-1))
        self.log("val loss", loss, prog_bar=True, on_step=True, on_epoch=True)
        self.log("val metrics", val_metrics_batch, prog_bar=True, on_step=False, on_epoch=True)
        self.metrices.reset()

    def predict_step(self, batch_idx, feature, dataloader_idx=0):
        logits = self.classifier(self.model(feature))
        logits_indices = torch.argmax(logits, dim=-1)
        return logits_indices

    def configure_optimizers(self) -> Any:
        return torch.optim.Adam(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)


def parse_args():
    args = argparse.ArgumentParser()
    args.add_argument('-p', '--ckpt', type=str)
    args.add_argument('-t', '--train', type=str, default="train")
    args.add_argument('-d', "--dataset", type=str, default="cola")
    args.add_argument('-s', "--text", type=str, default="hello word")
    return args.parse_args()


if __name__ == "__main__":
    args = parse_args()
    modelConfigs = OmegaConf.load("configs/model/proformer.yml")
    dataConfigs = OmegaConf.load("configs/dataset/cola.yml".format(args.dataset))
    model = ModulePLStyle(num_classes=dataConfigs.num_classes, **modelConfigs)
    data = DataModules.DataModule(**dataConfigs)
    if args.ckpt is not None:
        model.load_state_dict(args.ckpt)
    trainer = pl.Trainer(
        callbacks=[
            pl.callbacks.ModelCheckpoint(
                monitor='val metrics',
                filename='mixer-best-{epoch:03d}-{val metrics:.3f}',
                save_top_k=1,
                mode='max',
                save_last=True
            ),
        # early stopping
        # pl.callbacks.early_stopping.EarlyStopping(
        #     monitor="val_acc",
        #     min_delta=0.001,
        #     mode='max'
        # )
        ],
        enable_checkpointing=True,
        accelerator='cpu',
        devices=1,
        log_every_n_steps=1,
        logger=pl.loggers.TensorBoardLogger("logs/", args.dataset),
        max_epochs=8,
        check_val_every_n_epoch=1,
        # limit_train_batches=0.5,
        # limit_val_batches=0.1
    )
    if args.train == 'train':
        trainer.fit(model, data)
    if args.train == 'test':
        trainer.test(model, data, ckpt_path=args.ckpt)
    if args.train == "predict":
        trainer.predict(model, data, ckpt_path=args.ckpt)
