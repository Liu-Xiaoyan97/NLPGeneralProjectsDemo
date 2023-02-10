from pathlib import Path
import pandas as pd
import json
import re
"""
本文件将完成常用数据存储类型读取，包括txt,csv,tsv,xml,json等
"""


def FileReader(path: str):
    data = pd.read_csv(path)
    return data


def CSVReader(path: str):
    data = pd.read_csv(path)
    return data


def TSVReader(path: str):
    data = pd.read_csv(path, delimiter='\t')
    return data


def JsonReader(path: str):
    '''
    pip install lxml
    :param path:
    :return:
    '''
    data = pd.read_json(path)
    return data


def XmlReader(path: str):
    data = pd.read_xml(path)
    return data
