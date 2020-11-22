"""
每个子任务写为一个类
TriggerExtractor刚写了个开头
下面的维度还没仔细想，有可能是错的，model想清楚维度最重要
可以参考：reference\xf_event_extraction2020Top1-master\src_final\utils\model_utils.py
"""

import os
import torch
import torch.nn as nn
from torchcrf import CRF
from transformers import BertModel
from config import Config



# trigger 提取器
class TriggerExtractor(nn.Module):
    def __init__(self):
        super(Model, self).__init__()   

        # BERT层
        self.bert_model = BertModel.from_pretrained(Config.bert_dir)
        self.bert_config = self.bert_model.config
        for param in self.bert.parameters():
            param.requires_grad = True #微调时是否调BERT，True的话就调

        out_dims = self.bert_config.hidden_size
        mid_linear_dims=128

        # 加个线性层
        self.mid_linear = nn.Sequential(
            nn.Linear(out_dims, mid_linear_dims),
            nn.ReLU(),
            nn.Dropout(0.1)
        )

        # 分类器
        self.classifier = nn.Linear(mid_linear_dims, 2)

        # 激活函数
        self.activation = nn.Sigmoid()

        # 用于二分类的交叉熵损失函数
        self.criterion = nn.BCELoss()



    def forward(self, input_tensor, attention_mask=None):
        #attention_mask用于微调BERT，是对padding部分进行mask
        embedding_output, _ = self.bert(input_ids, attention_mask=attention_mask)  #shape:(batch_size, sequence_length, 768)