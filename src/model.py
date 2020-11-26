import os
import torch.nn as nn
from transformers import BertModel
from config import Config


# trigger 提取器
class TriggerExtractor(nn.Module):
    def __init__(self):
        super(TriggerExtractor, self).__init__()   

        # BERT层
        self.bert_model = BertModel.from_pretrained(Config.bert_dir)
        self.bert_config = self.bert_model.config
        for param in self.bert_model.parameters():
            param.requires_grad = True # 微调时是否调BERT，True的话就调

        # 线性层
        self.mid_linear = nn.Sequential(
            # nn.Linear要求输入是二维的，而bert输出是三维的，这里舍弃batch_size的那个维度，我也不知道为啥
            nn.Linear(self.bert_config.hidden_size, Config.trigger_txtractor_mid_linear_dims), 
            nn.ReLU(),
            nn.Dropout(0.1)
        )
 
        # 分类器
        self.classifier = nn.Linear(Config.trigger_txtractor_mid_linear_dims, 2)

        # 激活函数
        self.activation = nn.Sigmoid()

        # 用于二分类的交叉熵损失函数
        self.criterion = nn.BCELoss()



    def forward(self, input_tensor, attention_mask=None):
        # attention_mask用于微调BERT，是对padding部分进行mask
        # shape:(batch_size, sequence_length, hidden_size) hidden_size=768，就是这个bert的参数
        embedding_output, _ = self.bert_model(input_tensor, attention_mask=attention_mask)


        return embedding_output



