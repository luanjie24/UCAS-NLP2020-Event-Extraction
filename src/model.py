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
            # nn.Linear要求输入是二维的，而bert输出是三维的，这里舍弃batch_size的那个维度
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
        # shape:(batch_size, Config.sequence_length, hidden_size) hidden_size=768，就是这个bert的参数
        embedding_output, _ = self.bert_model(input_tensor, attention_mask=attention_mask)

        # 这块貌似就取第一句话的特征，这样输入batch个句子最后只输出一个句子的结果
        seq_out = embedding_output[0,:,:] # shape:(Config.sequence_length, hidden_size) 

        # 将输出的embedding融合触发词特征
        # if self.use_distant_trigger:
        #     assert distant_trigger is not None, \
        #         'When using distant trigger features, distant trigger should be implemented'

        #     distant_trigger_feature = self.distant_trigger_embedding(distant_trigger)
        #     seq_out = torch.cat([seq_out, distant_trigger_feature], dim=-1)

        seq_out = self.mid_linear(seq_out) # shape:(Config.sequence_length, Config.trigger_txtractor_mid_linear_dims)

        # 分类器 2列，第一列表示这个字是不是trigger的起始字，第二列表示这个字是不是trigger的终止字
        logits = self.activation(self.classifier(seq_out)) # shape:(Config.sequence_length, 2)

        # out = (logits,)

        return logits



