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

        # out_dims = self.bert_config.hidden_size
        # mid_linear_dims=128

        # # 加个线性层
        # self.mid_linear = nn.Sequential(
        #     nn.Linear(out_dims, mid_linear_dims),
        #     nn.ReLU(),
        #     nn.Dropout(0.1)
        # )
 
        # # 分类器
        # self.classifier = nn.Linear(mid_linear_dims, 2)

        # # 激活函数
        # self.activation = nn.Sigmoid()

        # # 用于二分类的交叉熵损失函数
        # self.criterion = nn.BCELoss()



    def forward(self, input_tensor, attention_mask=None):
        # attention_mask用于微调BERT，是对padding部分进行mask
        # shape:(batch_size, sequence_length, hidden_size) hidden_size=768，就是这个bert的参数
        embedding_output, _ = self.bert_model(input_tensor, attention_mask=attention_mask)
        
          
        return embedding_output




# import torch.nn as nn
# import torch.nn.functional as F
# from pytorch_transformers import BertModel


# class fn_cls(nn.Module):
#     def __init__(self):
#         super(fn_cls, self).__init__()
#         self.model = BertModel.from_pretrained(model_name, cache_dir="./")
#         self.model.to(device)
#         self.dropout = nn.Dropout(0.1)
#         self.l1 = nn.Linear(768, 2)

#     def forward(self, x, attention_mask=None):
#         outputs = self.model(x, attention_mask=attention_mask)
#         x = outputs[1]  # 取池化后的结果 batch * 768
#         x = x.view(-1, 768)
#         x = self.dropout(x)
#         x = self.l1(x)
#         return x