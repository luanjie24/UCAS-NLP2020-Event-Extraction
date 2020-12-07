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
import torch.nn as nn
import train
from config import Config
from model import TriggerExtractor, SubObjExtractor, TimeLocExtractor
from load_data import CorpusData
import argparse


def main(num: int):
  

    # if torch.cuda.device_count() > 1:
    #     print("Let's use", torch.cuda.device_count(), "GPUs!")
    #     # dim = 0 [30, xxx] -> [10, ...], [10, ...], [10, ...] on 3 GPUs
    #     trigger_extractor = TriggerExtractor()
    #     trigger_extractor = nn.DataParallel(TriggerExtractor)
    #     trigger_extractor.to(Config.device)

    corpus_data = CorpusData.load_corpus_data(Config.dataset_train)
    input_ids = torch.from_numpy(corpus_data["input_ids"]).long().to(Config.device)
    attention_masks = torch.from_numpy(corpus_data["attention_masks"]).long().to(Config.device)
    trigger_labels = torch.from_numpy(corpus_data["trigger_labels"]).long().to(Config.device)
    sub_obj_labels = torch.from_numpy(corpus_data["sub_obj_labels"]).long().to(Config.device)
    time_loc_labels = torch.from_numpy(corpus_data["time_loc_labels"]).long().to(Config.device)
    trigger_index = torch.from_numpy(corpus_data["trigger_index"]).long().to(Config.device)

    if num==1:
        train_dataset = TensorDataset(input_ids, attention_masks, trigger_labels)
        trigger_extractor = TriggerExtractor() 
        # 如果有多显卡
        # https://pytorch.org/tutorials/beginner/blitz/data_parallel_tutorial.html
        # 
        if torch.cuda.device_count() > 1:
            print("Let's use", torch.cuda.device_count(), "GPUs!")
            # dim = 0 [30, xxx] -> [10, ...], [10, ...], [10, ...] on 3 GPUs
            trigger_extractor = nn.DataParallel(trigger_extractor)

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
        raise ValueError("编号错误")



# 这里解析命令行参数
# usage
# python main -h 
# python main -i num # num为模型标号
if __name__ == '__main__':

    # help_str:str =  "Model Index" + "."*10 + "Model Name\n" \
    #                 + "  1" + "."*14 +  "TriggerExtractor\n" \
    #                 +  "  2" + "."*14 +  "SubObjExtractor\n" \
    #                 + "  3" + "."*14 +  "TimeLocExtractor"
    # num = int(input("输入训练模型编号："))
    # 模型的编号
    arg_range = range(1,4)
    parser = argparse.ArgumentParser(description='Choose One Model for Training')
    parser.add_argument("-i", type=int,  choices=arg_range, 
                        help='1 for TriggerExtractor, 2 for SubObjExtractor, 3 for TimeLocExtractor')

    args = parser.parse_args()
    if args.i not in arg_range:
        raise ValueError("编号错误")

    main(args.i)
    
