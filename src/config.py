import torch
import os, sys

class Config:
    os.chdir(sys.path[0])#这句话时防止相对路径在VSCode里用不了
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu') #看有没有gpu(cuda)，没有就用cpu

    # 数据集的路径 
    dataset_raw = 'nlp/dataset/CEC-Corpus/raw corpus'
    dataset_path = 'nlp/dataset/CEC-Corpus/CEC' 
    dataset_clean_path = 'nlp/dataset/CEC-Corpus/CEC_clean' # 数据预处理后的数据存在这里

    # 预训练模型所在路径
    bert_dir='../pretrained_model/chinese_roberta_wwm_ext_pytorch'

    # 超参数
    batch_size = 2 # 现在设置为2是为了测试
    sequence_length = 256 # 每句话的长度，不到这个长度就padding
    trigger_txtractor_mid_linear_dims = 128 # 这个是触发词提取模型的线性层的输出维度


