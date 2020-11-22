import torch

class Config:
    device = torch.device('cuda: 3' if torch.cuda.is_available() else 'cpu') #看有没有cuda，没有就只能用cpu了

    # 数据集的路径 
    dataset_raw = 'nlp/dataset/CEC-Corpus/raw corpus'
    dataset_path = 'nlp/dataset/CEC-Corpus/CEC' 
    dataset_clean_path = 'nlp/dataset/CEC-Corpus/CEC_clean' # 数据预处理后的数据存在这里

    # 预训练模型所在路径
    bert_dir='nlp/pretrained_model/chinese_roberta_wwm_ext_pytorch'

