import datasets
from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader
from transformers import BertTokenizer


class DataModule(LightningDataModule):

    def __init__(self, dataset: str, **kwargs):
        super(DataModule, self).__init__(**kwargs)
        self.dataset = dataset
        self.tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
        self.tokenizer.add_special_tokens({'pad_token': '[PAD]'})

    def get_dataset_cls(self):
        if self.dataset == 'imdb':
            return ImdbDataset
        if self.dataset == 'sst2':
            return SST2Dataset
        if self.dataset == 'agnews':
            return AGDataset
        if self.dataset == 'snli':
            return SNLIDataset
        if self.dataset == 'qnli':
            return QNLIDataset
        if self.dataset == 'yelp2':
            return YelpDataset
        if self.dataset == 'qqp':
            return QQPDataset
        if self.dataset == 'rte':
            return RTEDataset
        if self.dataset == 'cola':
            return CoLADataset
        if self.dataset == "hyperpartisan":
            return HyperpartisanDataset
        if self.dataset == "amazon":
            return AmazonDataset
        if self.dataset == "dbpedia":
            return dbpediaDataset

    def setup(self, stage: str = None):
        dataset_cls = self.get_dataset_cls()
        print(dataset_cls)
        if stage in (None, 'fit'):
            self.train_set = dataset_cls('train', self.datasetcfg.dataset_type.max_seq_len, self.tokenizer,
                                         self.projecion, self.label_map)
            if self.dataset in ['imdb', "yelp2", 'agnews', 'dbpedia', "amazon"]:
                mode = 'test'
            else:
                mode = 'validation'
            self.eval_set = dataset_cls(mode, self.datasetcfg.dataset_type.max_seq_len, self.tokenizer, self.projecion,
                                        self.label_map)
        if stage in (None, 'test'):
            if self.dataset in ['imdb', "yelp2", 'agnews', 'dbpedia', "amazon"]:
                mode = 'test'
            else:
                mode = 'validation'
            self.test_set = dataset_cls(mode, self.datasetcfg.dataset_type.max_seq_len, self.tokenizer, self.projecion,
                                        self.label_map)

    def train_dataloader(self) -> DataLoader:
        return DataLoader(self.train_set, self.modelcfg.loader.batch_size, shuffle=True,
                          num_workers=self.modelcfg.loader.num_workers, persistent_workers=True)

    def val_dataloader(self) -> DataLoader:
        return DataLoader(self.eval_set, self.modelcfg.loader.batch_size, shuffle=True,
                          num_workers=self.modelcfg.loader.num_workers, persistent_workers=True)

    def test_dataloader(self) -> DataLoader:
        return DataLoader(self.test_set, self.modelcfg.loader.batch_size, shuffle=True,
                          num_workers=self.modelcfg.loader.num_workers, persistent_workers=True)
    
