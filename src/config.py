import torch
import os, sys

class Config:
    os.chdir(sys.path[0])# 这句话时防止相对路径在VSCode里用不了
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # 数据集的路径 
    # dataset_train = '../dataset/xf_2020_Corpus/final/raw_data/preliminary_data_pred_trigger_and_role.json' 
    # dataset_saved = '../saved_model/dev_data' # 数据预处理后的数据存在这里
    # 训练集
    dataset_train = '../dataset/xf_2020_Corpus/final/EE2020/train.json'
    dataset_dev = '../dataset/xf_2020_Corpus/final/EE2020/dev.json'
    # 测试集
    dataset_test = '../dataset/xf_2020_Corpus/final/EE2020/test.json'

    # 预训练模型所在路径
    bert_dir='../pretrained_model/chinese_roberta_wwm_ext_pytorch'
    bert_cache_dir = '../saved_model/bert_cached'
    # 训练完模型存储路径
    saved_trigger_extractor_dir = '../saved_model/models/TriggerExtractor'
    saved_sub_obj_extractor_dir = '../saved_model/models/SubObjExtractor'
    saved_time_loc_extractor_dir = '../saved_model/models/TimeLocExtractor'
    saved_best_trigger_extractor_dir = '../saved_model/models/TriggerExtractor/best_checkpoint'
    saved_best_sub_obj_extractor_dir = '../saved_model/models/SubObjExtractor/best_checkpoint'
    saved_best_time_loc_extractor_dir = '../saved_model/models/TimeLocExtractor/best_checkpoint'

    test_results_dir =  '../test_results'
    # 日志存储路径
    saved_log_dir='../saved_model/logs/train_info.log'
    # 参数
    train_batch_size = 25
    dev_batch_size = 25
    train_epochs = 40
    sequence_length = 256
    trigger_extractor_mid_linear_dims = 128
    sbj_obj_extractor_mid_linear_dims = 128
    time_loc_extractor_mid_linear_dims = 128
    weight_decay = 0.0001
    bert_learning_rate = 0.0000001
    other_learning_rate = 0.00006
    adam_epsilon = 0.00000001
    warmup_proportion = 0.1
    max_grad_norm = 1.0
    num_workers = 0

    start_threshold = 0.1
    end_threshold = 1
    
    version = 1.0

