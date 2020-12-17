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

        self.bert_model = BertModel.from_pretrained(Config.bert_dir, cache_dir=Config.bert_cache_dir)
        self.bert_model.eval()
        self.bert_config = self.bert_model.config
        for param in self.bert_model.parameters():
            param.requires_grad = True

        self.mid_linear = nn.Sequential(
            nn.Linear(self.bert_config.hidden_size, Config.trigger_extractor_mid_linear_dims), 
            nn.ReLU(),
            nn.Dropout(0.1)
        )
 
        self.classifier = nn.Linear(Config.trigger_extractor_mid_linear_dims, 2)

        self.activation = nn.Sigmoid()

        self.criterion = nn.BCELoss()


    def forward(self, input_tensor, attention_mask=None, labels=None):

        embedding_output =self.bert_model(input_tensor, attention_mask=attention_mask)

        seq_out = self.mid_linear(embedding_output[0])

        logits = self.activation(self.classifier(seq_out))

        out = (logits,)

        if labels is not None:
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


# sub & obj 提取器
class SubObjExtractor(nn.Module):
    def __init__(self):
        super(SubObjExtractor, self).__init__()

        self.bert_model = BertModel.from_pretrained(Config.bert_dir, cache_dir=Config.bert_cache_dir)
        self.bert_config = self.bert_model.config
        for param in self.bert_model.parameters():
            param.requires_grad = True 

        # Conditional Layer Normalization层 layer_norm_eps没这个参数啊
        self.conditional_layer_norm = ConditionalLayerNorm(self.bert_config.hidden_size, eps=self.bert_config.layer_norm_eps)

        self.mid_linear = nn.Sequential(
            nn.Linear(self.bert_config.hidden_size, Config.sbj_obj_extractor_mid_linear_dims),
            nn.ReLU(),
            nn.Dropout(0.1)
        )

        self.obj_classifier = nn.Linear(Config.sbj_obj_extractor_mid_linear_dims, 2)
        self.sub_classifier = nn.Linear(Config.sbj_obj_extractor_mid_linear_dims, 2)

        self.activation = nn.Sigmoid()

        self.criterion = nn.BCELoss()

    def forward(self, input_tensor, trigger_index, attention_mask,  labels=None):

        embedding_output = self.bert_model(input_tensor, attention_mask=attention_mask) 
       
        # 将embedding融合trigger的特征
        trigger_label_feature = self._batch_gather(embedding_output[0], trigger_index) 
        trigger_label_feature = trigger_label_feature.view([trigger_label_feature.size()[0], -1]) 
        seq_out = self.conditional_layer_norm(embedding_output[0], trigger_label_feature)

        seq_out = self.mid_linear(seq_out) 
        obj_logits = self.activation(self.obj_classifier(seq_out))
        sub_logits = self.activation(self.sub_classifier(seq_out))


        logits = torch.cat([obj_logits, sub_logits], dim=-1)
        out = (logits,)

        if labels is not None:
            masks = torch.unsqueeze(attention_mask, -1)
            labels = labels.float()
            obj_loss = self.criterion(obj_logits * masks, labels[:, :, :2])
            sub_loss = self.criterion(sub_logits * masks, labels[:, :, 2:])
            loss = obj_loss + sub_loss
            out = (loss,) + out

        return out

    @staticmethod
    def _batch_gather(data: torch.Tensor, index: torch.Tensor):

        index = index.unsqueeze(-1).repeat_interleave(data.size()[-1], dim=-1)
        return torch.gather(data, 1, index)






# time & loc 提取器
class TimeLocExtractor(nn.Module):
    def __init__(self):
        super(TimeLocExtractor, self).__init__()

        self.bert_model = BertModel.from_pretrained(Config.bert_dir, cache_dir=Config.bert_cache_dir)
        self.bert_config = self.bert_model.config
        for param in self.bert_model.parameters():
            param.requires_grad = True

        # Conditional Layer Normalization层 
        self.conditional_layer_norm = ConditionalLayerNorm(self.bert_config.hidden_size, eps=self.bert_config.layer_norm_eps)

        self.mid_linear = nn.Sequential(
            nn.Linear(self.bert_config.hidden_size, Config.sbj_obj_extractor_mid_linear_dims),
            nn.ReLU(),
            nn.Dropout(0.1)
        )

        self.classifier = nn.Linear(Config.time_loc_extractor_mid_linear_dims, 10)

        self.crf_module = CRF(num_tags=10, batch_first=True)



    def forward(self, input_tensor, trigger_index, attention_mask,  labels=None):

        embedding_output = self.bert_model(input_tensor, attention_mask=attention_mask) 

        # 将embedding融合trigger的特征
        trigger_label_feature = self._batch_gather(embedding_output[0], trigger_index)
        trigger_label_feature = trigger_label_feature.view([trigger_label_feature.size()[0], -1])
        # 放到Conditional Layer Normalization层
        seq_out = self.conditional_layer_norm(embedding_output[0], trigger_label_feature)

        seq_out = self.mid_linear(seq_out)

        #seq_out = self.mid_linear(embedding_output[0])

        emissions = self.classifier(seq_out)
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