import torch
from torch.utils.data import DataLoader, Dataset
from transformers import BertTokenizer
import pandas as pd
'''
本文件将数据读取和过滤字符及分词合并为DataPreProcess类，直接调用DataPreProcess.data可得到数据
example:
path = file_path
Data = DataPreProcess(path).data
'''


class DataPreProcess:
    def __init__(self, path: str):
        self.path = path
        self.tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
        self.tokenizer.add_special_tokens({'pad_token': '[PAD]'})
        self.data = self.autoprocess()

    def FileReader(self):
        with open(self.path, "r", encoding="utf-8") as f:
            data = f.read().splitlines()
        text = []
        label = []
        for row in data:
            tmp = row.split("\t")
            text.append(tmp[0])
            label.append(tmp[1])
        data = {"Text": text[1:], "Label": label[1:]}
        del text, label
        data = pd.DataFrame(data, columns=["Text", "Label"])
        return data

    def CSVReader(self):
        data = pd.read_csv(self.path)
        return data

    def TSVReader(self):
        data = pd.read_csv(self.path, delimiter='\t')
        return data

    def JsonReader(self):
        '''
        pip install lxml
        :param path:
        :return:
        '''
        data = pd.read_json(self.path)
        return data

    def XmlReader(self):
        data = pd.read_xml(self.path)
        return data

    @staticmethod
    def Normalization(text: str):
        return text.replace("!", "") \
            .replace("@", "") \
            .replace("#", "") \
            .replace("$", "") \
            .replace("%", "") \
            .replace("/", "") \
            .replace("\\", "")

    def PreProcess(self, data):
        for index, row in data.iterrows():
            data.at[index, 'Text'] = self.Normalization(row["Text"])
        return data

    def ReaderChoice(self, file_type):
        if file_type == "txt":
            return self.FileReader()
        if file_type == "csv":
            return self.CSVReader()
        if file_type == "tsv":
            return self.TSVReader()
        if file_type == "json":
            return self.JsonReader()
        if file_type == "xml":
            return self.XmlReader()

    def autoprocess(self):
        file_type = self.path[-5:].split('.')[1]
        data = self.PreProcess(self.ReaderChoice(file_type))
        text = data["Text"].values.tolist()
        labels = data["Label"].values.tolist()
        del data
        tokens = self.tokenizer.__call__(text=text, padding=True, truncation=True, max_length=128)
        return {
            "tokens": tokens,
            "labels": labels
        }


class OwnDataset(Dataset):
    def __init__(self, path: str, **kwargs):
        super(OwnDataset, self).__init__(**kwargs)
        self.data = DataPreProcess(path).data

    def __len__(self):
        return len(self.data["tokens"]['input_ids'])

    def __getitem__(self, item):
        return {
            "tokens": {
                'input_ids': torch.Tensor(self.data["tokens"]['input_ids'][item]).long(),
                'token_type_ids': torch.Tensor(self.data["tokens"]['token_type_ids'][item]).long(),
                'attention_mask': torch.Tensor(self.data['tokens']['attention_mask'][item]).long()
            },
            "labels": self.data["labels"][item]
        }
    
