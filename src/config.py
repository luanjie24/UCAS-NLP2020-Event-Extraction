import torch
import os, sys

class Config:
    os.chdir(sys.path[0])# 这句话时防止相对路径在VSCode里用不了
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu') # 看有没有gpu(cuda)，没有就用cpu

    # 数据集的路径 
    dataset_train = '../dataset/xf_2020_Corpus/preliminary/raw_data/dev.json' # 总共576条数据
    dataset_saved = '../saved_model/dev_data' # 数据预处理后的数据存在这里

    # 预训练模型所在路径
    bert_dir='../pretrained_model/chinese_roberta_wwm_ext_pytorch'
    # 训练完模型存储路径
    saved_trigger_extractor_dir = '../saved_model/models/TriggerExtractor'
    saved_sub_obj_extractor_dir = '../saved_model/models/SubObjExtractor'
    # 日志存储路径
    saved_log_dir='../saved_model/logs/train_info.log'
    # 参数
    train_batch_size = 2
    train_epochs = 1
    sequence_length = 10 # 每句话的长度，不到这个长度就padding
    trigger_extractor_mid_linear_dims = 128 # 这个是触发词提取模型的线性层的输出维度
    sbj_obj_extractor_mid_linear_dims = 128 # 这个是主客体识别模型的线性层的输出维度
    weight_decay = 0.0001 # optimizer的权重衰减，概念参见L2正则化
    bert_learning_rate = 0.0000001 # bert模型的学习率
    other_learning_rate = 0.00001 # 其他模型的学习率
    adam_epsilon = 0.00000001 # 这个是优化器为了增加数值计算的稳定性而加到分母里的项，默认就是1e-8，可以先不改
    warmup_proportion = 0.1 # scheduler慢热学习的比例，貌似用0.的1比较多
    max_grad_norm = 1.0 # 大于1的梯度将其设为1.0, 以防梯度爆炸
    


