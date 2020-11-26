"""
train.py用于训练集的训练

"""

from config import Config
from model import TriggerExtractor
import torch


if __name__ == '__main__':

    # 读取数据并进行预处理，load_data建议写成一个类（也就是以下内容都应该在load_data.py或者data_converter.py的类里以供这里调用）
    # 这里先拿2句话当作输入做测试，即batch_size=2，padding和mask这里先不做，但最终预处理的时候需要包含
    test_samples1 = '昨天清华着火了！'
    test_samples2 = '清华大学真厉害。'
    from transformers import BertTokenizer
    tokenizer = BertTokenizer.from_pretrained(Config.bert_dir)

    # input_ids就是一连串token在字典中的对应id。形状为(Config.batch_size, sequence_length)
    # sequence_length一般用padding把每句话的长度统一
    input_ids=[]
    input_ids.append(tokenizer.encode(test_samples1))
    input_ids.append(tokenizer.encode(test_samples2))
    input_tensor=torch.LongTensor(input_ids).cuda() # 本例中
    print(input_tensor) 

    # attention_mask用于指定对哪些词进行self-Attention操作，
    # 避免在padding的token上计算attention（1不进行masked，0则masked）。形状为(Config.batch_size, sequence_length)。
    # 如果不设置则BertModel默认全为1
    # batch_mask = torch.ones_like(input_tensor) 




    #==================
    # 初始化模型
    trigger_extractor = TriggerExtractor() 
    trigger_extractor.to(Config.device)
    output_tensor=trigger_extractor(input_tensor)#这里前向传播一次看看效果 这里应该再加一个attention_mask，目前为attention_mask=None
    print(output_tensor.shape)

