# NLPGeneralProjectsDemo
Author @ Xiaoyan Liu
## 你可以在本项目中学到什么?
- NLP代码的前期工作（包括文件读取、文本预处理、分词、文本静态嵌入等）
- 神经网络的搭建（包括动态词嵌入、神经网络的搭建、分类器的使用等）
- 神经网络的评估指标使用
- 高级辅助框架 Pytorch-lightning的使用
- 更多如图卷积等将持续更新
## 项目树

## 如何查看训练日志
将下列命令输入到cmd中执行，--logdir替换为自己的目录就行。以事件文件events.out.tfevents.xxx.lxy为例：  
```Tensorboard --logdir=logs/TextCNN/version_0```
## 如何运行
MODE: train/test/predict三者中任一    
CKPT_PATH: 模型存放位置，用来加载训练好的模型  
DATASET_PATH: 数据存放位置，用来加载数据  
PREDICT_TEXT: 待预测文本  
```python main.py -t MODE -p CKPT_PATH -d DATASET_PATH -s PREDICT_TEXT```  
example:  
```python main.py -t test -p logs/TextCN/version_0/model.pth, -d Data/1.json -s "hello world"```