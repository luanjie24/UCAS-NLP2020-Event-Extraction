"""
main.py目前是对train.py的进一步封装，方便训练
训练时的一些注意点：
    1.调参的参数都在Config.py中，可以认真读一下
    2.我的PC只有一个GPU，所以目前的实验是单GPU训练，如果改为多GPU可参考https://blog.csdn.net/daydayjump/article/details/81158777
    3.包括调参和多GPU训练在内的代码改动尽量只改动config.py，如果要改其他文件可以先和我说一下
    4.训练集还可以扩充
    5.模型1预处理部分有bug，模型2，3预处理部分还没写完

"""

import torch
from torch.utils.data import DataLoader, RandomSampler, TensorDataset
import train
from config import Config
from model import TriggerExtractor, SubObjExtractor, TimeLocExtractor
from load_data import CorpusData


def main():
    print("模型编号","."*10, "模型名称")
    print("  1","."*14, "TriggerExtractor")
    print("  2","."*14, "SubObjExtractor")
    print("  3","."*14, "TimeLocExtractor")
    num = int(input("输入训练模型编号："))

    corpus_data = CorpusData.load_corpus_data(Config.dataset_train)
    input_ids = torch.from_numpy(corpus_data["input_ids"]).long() 
    attention_masks = torch.from_numpy(corpus_data["attention_masks"]).long() 
    trigger_labels = torch.from_numpy(corpus_data["trigger_labels"]).long() 

    if num==1:
        train_dataset = TensorDataset(input_ids, attention_masks, trigger_labels)
        trigger_extractor = TriggerExtractor() 
        trigger_extractor.to(Config.device)
        train.train(trigger_extractor, train_dataset, Config.saved_trigger_extractor_dir)
    elif num==2:
        train_dataset = TensorDataset(input_ids, trigger_index, attention_masks, sub_obj_labels)
        sub_obj_extractor = SubObjExtractor()
        sub_obj_extractor.to(Config.device)
        train.train(sub_obj_extractor, train_dataset, Config.saved_sub_obj_extractor_dir)
    elif num==3:
        train_dataset = TensorDataset(input_ids, trigger_index, attention_masks, time_loc_labels)
        time_loc_extractor = TimeLocExtractor()
        time_loc_extractor.to(Config.device)
        train.train(time_loc_extractor, train_dataset, Config.saved_time_loc_extractor_dir)
    else:
        print("编号错误")
    





if __name__ == '__main__':

    main()
    
