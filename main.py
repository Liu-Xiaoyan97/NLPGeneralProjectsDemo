import argparse
from Modules import TextCNN
from omegaconf import OmegaConf


def parse_args():
    args = argparse.ArgumentParser()
    args.add_argument('-p', '--ckpt', type=str)
    args.add_argument('-t', '--train', type=str, default="train")
    args.add_argument('-d', "--dataset", type=str, default="./Data/1.json")
    args.add_argument('-s', "--text", type=str, default="hello word")
    return args.parse_args()


if __name__ == "__main__":
    configs = OmegaConf.load("./config/textcnn.yml")
    args = parse_args()
    print(configs)
    # num_embeddings=100000
    # embedding_dim=128
    # max_seq_len=10
    # in_channels=1
    # out_channels=1
    # kernels=[{'kernel_size': 7, "padding": 3, "stride": 1},
    #          {'kernel_size': 5, 'padding': 2, 'stride': 1},
    #          {'kernel_size': 3, 'padding': 1, 'stride': 1},
    #          {'kernel_size': 7, 'padding': 3, 'stride': 1}]
    # num_classes=4
    # lr=1e-4
    # weight_decay=1e-3
    model = TextCNN(**configs)
    # model.trainepoch("./logs/", "./Data/1.json")
    if args.train == "train":
        model.trainepoch("logs/", args.dataset, 10, 3)
    if args.train == "test":
        model.test("logs/", args.dataset, args.ckpt, 10)
    if args.train == "predict":
        p = model.predict(args.text, args.ckpt)
        print(p)