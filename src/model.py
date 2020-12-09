import os
import torch
import torch.nn as nn
from torchcrf import CRF
from transformers import BertModel
from config import Config

# trigger 提取器
class TriggerExtractor(nn.Module):
    def __init__(self):
        super(TriggerExtractor, self).__init__()   

        # BERT层
        self.bert_model = BertModel.from_pretrained(Config.bert_dir, cache_dir=Config.bert_cache_dir)
        self.bert_config = self.bert_model.config
        for param in self.bert_model.parameters():
            param.requires_grad = True # 微调时是否调BERT，True的话就调

        '''
        # 将输出的embedding融合触发词特征，即distant_trigger相关，这个先不做了
        if use_distant_trigger:
            embedding_dim = kwargs.pop('embedding_dims', 256)
            self.distant_trigger_embedding = nn.Embedding(num_embeddings=2, embedding_dim=embedding_dim)
            out_dims += embedding_dim
        '''

        # 线性层
        self.mid_linear = nn.Sequential(
            nn.Linear(self.bert_config.hidden_size, Config.trigger_extractor_mid_linear_dims), 
            nn.ReLU(),
            nn.Dropout(0.1)
        )
 
        # 分类器
        self.classifier = nn.Linear(Config.trigger_extractor_mid_linear_dims, 2)

        # 激活函数
        self.activation = nn.Sigmoid()

        # 用于二分类的交叉熵损失函数
        self.criterion = nn.BCELoss()

        '''
        # distant_trigger相关，先不做了
        # init_blocks = [self.mid_linear, self.classifier]
        # if use_distant_trigger:
        #     init_blocks += [self.distant_trigger_embedding]
        '''

    def forward(self, input_tensor, attention_mask=None, labels=None):
        # input_tensor shape: (Config.train_batch_size, Config.sequence_length)
        # attention_mask shape:(Config.train_batch_size, Config.sequence_length)
        embedding_output, _ = self.bert_model(input_tensor, attention_mask=attention_mask)

        # 这块貌似就取第一句话的特征，这样输入batch个句子最后只输出一个句子的结果，不知道参考代码为什么这么做，舍弃了
        # seq_out = embedding_output[0,:,:] # shape:(Config.sequence_length, hidden_size) 
        # print(embedding_output[0,:,:].shape)
        # print(embedding_output[:,0,:].shape) # shape:(Config.train_batch_size, bert_hidden_size)

        '''
        # distant_trigger相关，先不做了
        if self.use_distant_trigger:
            assert distant_trigger is not None, \
                'When using distant trigger features, distant trigger should be implemented'
            distant_trigger_feature = self.distant_trigger_embedding(distant_trigger)
            seq_out = torch.cat([seq_out, distant_trigger_feature], dim=-1)
        '''

        seq_out = self.mid_linear(embedding_output) # shape:(Config.train_batch_size, Config.sequence_length, Config.trigger_extractor_mid_linear_dims)
        # print(seq_out.shape) 

        # 分类器 2列，第一列表示这个字是不是trigger的起始字，第二列表示这个字是不是trigger的终止字
        logits = self.activation(self.classifier(seq_out)) # shape:(Config.train_batch_size, Config.sequence_length, 2)

        out = (logits,)

        # 算损失
        if labels is not None:
            # 这块也可以像第二个模型一样乘一个attentionmask减少padding部分的影响
            loss = self.criterion(logits, labels.float())
            out = (loss,) + out

        return out


# 论元提取子结构，用ConditionalLayerNorm让文本融入Trigger的语义信息 https://kexue.fm/archives/7124
class ConditionalLayerNorm(nn.Module):
    def __init__(self,
                 normalized_shape,
                 eps=1e-12):
        super().__init__()

        self.eps = eps

        self.weight = nn.Parameter(torch.Tensor(normalized_shape))
        self.bias = nn.Parameter(torch.Tensor(normalized_shape))

        self.weight_dense = nn.Linear(normalized_shape * 2, normalized_shape, bias=False)
        self.bias_dense = nn.Linear(normalized_shape * 2, normalized_shape, bias=False)

        self.reset_weight_and_bias()

    def reset_weight_and_bias(self):
        """
        此处初始化的作用是在训练开始阶段不让 conditional layer norm 起作用
        """
        nn.init.ones_(self.weight)
        nn.init.zeros_(self.bias)

        nn.init.zeros_(self.weight_dense.weight)
        nn.init.zeros_(self.bias_dense.weight)

    def forward(self, inputs, cond=None): #(Config.train_batch_size, Config.sequence_length, bert_hidden_size) # shape (Config.train_batch_size, -1)
        assert cond is not None, 'Conditional tensor need to input when use conditional layer norm'
        cond = torch.unsqueeze(cond, 1)  # (b, 1, h*2)

        weight = self.weight_dense(cond) + self.weight  # (b, 1, h)
        bias = self.bias_dense(cond) + self.bias  # (b, 1, h)

        mean = torch.mean(inputs, dim=-1, keepdim=True)  # （b, s, 1）
        outputs = inputs - mean  # (b, s, h)

        variance = torch.mean(outputs ** 2, dim=-1, keepdim=True)
        std = torch.sqrt(variance + self.eps)  # (b, s, 1)

        outputs = outputs / std  # (b, s, h)

        outputs = outputs * weight + bias

        return outputs


# sub & obj 提取器 同样不考虑distant_trigger
class SubObjExtractor(nn.Module):
    def __init__(self):
        super(SubObjExtractor, self).__init__()

        # BERT层
        self.bert_model = BertModel.from_pretrained(Config.bert_dir, cache_dir=Config.bert_cache_dir)
        self.bert_config = self.bert_model.config
        for param in self.bert_model.parameters():
            param.requires_grad = True # 微调时是否调BERT，True的话就调

        # Conditional Layer Normalization层 layer_norm_eps没这个参数啊
        self.conditional_layer_norm = ConditionalLayerNorm(self.bert_config.hidden_size, eps=self.bert_config.layer_norm_eps)

        '''
        融合trigger特征这一层先跳过
        '''

        # 线性层
        self.mid_linear = nn.Sequential(
            nn.Linear(self.bert_config.hidden_size, Config.sbj_obj_extractor_mid_linear_dims),
            nn.ReLU(),
            nn.Dropout(0.1)
        )

        # 分类器
        self.obj_classifier = nn.Linear(Config.sbj_obj_extractor_mid_linear_dims, 2)
        self.sub_classifier = nn.Linear(Config.sbj_obj_extractor_mid_linear_dims, 2)

        # 激活函数
        self.activation = nn.Sigmoid()

        # 损失
        self.criterion = nn.BCELoss()

    def forward(self, input_tensor, trigger_index, attention_mask,  labels=None):
        # input_tensor shape: (Config.train_batch_size, Config.sequence_length)
        # embedding_output shape:(Config.train_batch_size, Config.sequence_length, bert_hidden_size)
        embedding_output, _ = self.bert_model(input_tensor, attention_mask=attention_mask) 
        # print(len(input_tensor),  len(trigger_index) )
        # print(trigger_index)
        # 将embedding融合trigger的特征
        # trigger_index shape:(Config.train_batch_size, n) trigger_index应该是trigger第一个字和最后一个字在文本中的位置（n应该永远等于2）
        trigger_label_feature = self._batch_gather(embedding_output, trigger_index) # shape (Config.train_batch_size, n, bert_hidden_size)
        trigger_label_feature = trigger_label_feature.view([trigger_label_feature.size()[0], -1]) # shape (Config.train_batch_size, n*bert_hidden_size) 
        # 放到Conditional Layer Normalization层
        # trigger_index shape:(Config.train_batch_size, Config.sequence_length, bert_hidden_size)
        seq_out = self.conditional_layer_norm(embedding_output, trigger_label_feature)

        '''
        融合trigger特征这一层先跳过
        '''

        seq_out = self.mid_linear(seq_out) # shape: (Config.train_batch_size, Config.sequence_length, Config.sbj_obj_extractor_mid_linear_dims)

        obj_logits = self.activation(self.obj_classifier(seq_out)) # shape: (Config.train_batch_size, Config.sequence_length, 2)
        sub_logits = self.activation(self.sub_classifier(seq_out)) # shape: (Config.train_batch_size, Config.sequence_length, 2)


        logits = torch.cat([obj_logits, sub_logits], dim=-1) # shape: (Config.train_batch_size, Config.sequence_length, 4)
        out = (logits,)

        if labels is not None:
            masks = torch.unsqueeze(attention_mask, -1) #shape:(Config.train_batch_size, Config.sequence_length, 1)
            labels = labels.float()
            obj_loss = self.criterion(obj_logits * masks, labels[:, :, :2])
            sub_loss = self.criterion(sub_logits * masks, labels[:, :, 2:])
            loss = obj_loss + sub_loss
            out = (loss,) + out

        return out

    @staticmethod
    def _batch_gather(data: torch.Tensor, index: torch.Tensor):
        """
        实现类似 tf.batch_gather 的效果
        :param data: (bs, max_seq_len, hidden)
        :param index: (bs, n)
        :return: a tensor which shape is (bs, n, hidden)
        """
        index = index.unsqueeze(-1).repeat_interleave(data.size()[-1], dim=-1)  # (bs, n, hidden)
        return torch.gather(data, 1, index)






# time & loc 提取器 同样不考虑distant_trigger 这个模型损使用CRF，因为time & loc 样本数量很少，同时随机丢弃70%的负样本，使正负样本均衡
class TimeLocExtractor(nn.Module):
    def __init__(self):
        super(TimeLocExtractor, self).__init__()

        # BERT层
        self.bert_model = BertModel.from_pretrained(Config.bert_dir, cache_dir=Config.bert_cache_dir)
        self.bert_config = self.bert_model.config
        for param in self.bert_model.parameters():
            param.requires_grad = True # 微调时是否调BERT，True的话就调

        # Conditional Layer Normalization层 layer_norm_eps没这个参数啊
        self.conditional_layer_norm = ConditionalLayerNorm(self.bert_config.hidden_size, eps=self.bert_config.layer_norm_eps)

        '''
        融合trigger特征这一层先跳过
        '''

        # 线性层
        self.mid_linear = nn.Sequential(
            nn.Linear(self.bert_config.hidden_size, Config.sbj_obj_extractor_mid_linear_dims),
            nn.ReLU(),
            nn.Dropout(0.1)
        )

        # 分类器
        self.classifier = nn.Linear(Config.time_loc_extractor_mid_linear_dims, 10)

        # CRF分类器 batch_first为true据说能让模型速度更快
        self.crf_module = CRF(num_tags=10, batch_first=True)



    def forward(self, input_tensor, trigger_index, attention_mask,  labels=None):
        # input_tensor shape: (Config.train_batch_size, Config.sequence_length)
        # embedding_output shape:(Config.train_batch_size, Config.sequence_length, bert_hidden_size)
        embedding_output, _ = self.bert_model(input_tensor, attention_mask=attention_mask) 

        # 将embedding融合trigger的特征
        # trigger_index shape:(Config.train_batch_size, n) trigger_index应该是trigger第一个字和最后一个字在文本中的位置（n应该永远等于2）
        trigger_label_feature = self._batch_gather(embedding_output, trigger_index) # shape (Config.train_batch_size, n, bert_hidden_size)
        trigger_label_feature = trigger_label_feature.view([trigger_label_feature.size()[0], -1]) # shape (Config.train_batch_size, n*bert_hidden_size) 
        # 放到Conditional Layer Normalization层
        # trigger_index shape:(Config.train_batch_size, Config.sequence_length, bert_hidden_size)
        seq_out = self.conditional_layer_norm(embedding_output, trigger_label_feature)

        '''
        融合trigger特征这一层先跳过
        '''

        seq_out = self.mid_linear(seq_out) # shape: (Config.train_batch_size, Config.sequence_length, Config.sbj_obj_extractor_mid_linear_dims)

        
        emissions = self.classifier(seq_out) # shape: (Config.train_batch_size, Config.sequence_length, 10) 
        if labels is not None:
            tokens_loss = -1. * self.crf_module(emissions=emissions,
                                                tags=labels.long(),
                                                mask=attention_mask.byte(),
                                                reduction='mean')

            out = (tokens_loss,)

        else:
            tokens_out = self.crf_module.decode(emissions=emissions, mask=attention_mask.byte())
            out = (tokens_out,)

        return out


    @staticmethod
    def _batch_gather(data: torch.Tensor, index: torch.Tensor):
        """
        实现类似 tf.batch_gather 的效果
        :param data: (bs, max_seq_len, hidden)
        :param index: (bs, n)
        :return: a tensor which shape is (bs, n, hidden)
        """
        index = index.unsqueeze(-1).repeat_interleave(data.size()[-1], dim=-1)  # (bs, n, hidden)
        return torch.gather(data, 1, index)