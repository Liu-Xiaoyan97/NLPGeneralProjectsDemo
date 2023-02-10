'''
本文件将完成字符过滤操作
'''
from FileReader import JsonReader


def Normalization(text: str):
    return text.replace("!", "")\
                .replace("@", "")\
                .replace("#", "")\
                .replace("$", "")\
                .replace("%", "")\
                .replace("/", "")\
                .replace("\\", "")


def PreProcess(data):
    for index, row in data.iterrows():
        data.at[index, 'Text'] = Normalization(row["Text"])
    return data






data = JsonReader("../Data/1.json")
PreProcess(data)