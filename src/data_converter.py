"""
data_converter用于读取所使用的公开数据集，并进行预处理（数据清洗或者说结构转化）

最好把每个子任务都写成一个类

肯定需要做的转换包括将输入转为input_id、加CLS和SEP标记、以及一些实验的数据集格式转化（比如拆分训练集测试集，再比如CEC数据集的标注太细了，需要转成合适的形式，还需要把标注内容格式化）
BIO标注我还没搞清楚对于roberta_wwm还有没有必要，因为它训练时已经是考虑词和字的区别了
padding和attention_mask应该是bert类模型都需要的，目前还没完全搞懂

有些地方还不是很确定，下面写的方法有可能会更改或者删除

参考：reference\xf_event_extraction2020Top1-master\src_final\preprocess\processor.py以及其他preprocess里的文件

目前所用数据集：https://github.com/shijiebei2009/CEC-Corpus
"""
import random
import os
import torch
from transformers import BertTokenizer
from config import Config
 

class DataConverter4Trigger:
    """
    例如这是针对第一个子任务（触发词识别）的数据预处理
    """
    def __init__(self):
        
        # 数据集文件夹路径
        self.dataset_dir = Config.dataset_path
        self.dataset_raw = Config.dataset_raw 
        # 读取数据集

        # 读Bert词汇表
        self.tokenizer = BertTokenizer.from_pretrained(Config.model_vocab_path)


    def load_tags(self):
        # 加载Trigger标签 这个得看看CEC数据集一共标记了几种Trigger，可以先人工写在一个txt文件里


    def set_BIO(self):
        # 给训练集和验证集打上BIO标注，这个可能不需要


    def save_BIO_data(self, data, save_dir):
        # 把打上BIO标注的data存起来
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        print("数据保存完毕")


    def set_input_ids(self):
        # 读取文件中的句子变成BERT需求的编码input_ids (tokenizer.encode()方法)


    def data_iterator(self):
        # padding 和 attention_mask相关的可写在这
        



