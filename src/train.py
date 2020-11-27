"""
train.py用于训练集的训练

"""

import torch
from config import Config
from model import TriggerExtractor
from transformers import AdamW, get_linear_schedule_with_warmup



def build_optimizer_and_scheduler(model, t_total):

    # 读取模型参数，模型可能来自于存起来的模型，也可能是来自于model.py初始化的模型，所以加了个下面的判断
    module = (
        model.module if hasattr(model, "module") else model
    )
    model_param = list(module.named_parameters())
    # print(model_param) # 不懂可以输出参数看看

    # 将bert的参数和其他参数分开
    bert_param_optimizer = []
    other_param_optimizer = []
    for name, para in model_param:
        space = name.split('.')
        if space[0] == 'bert_module':
            bert_param_optimizer.append((name, para))
        else:
            other_param_optimizer.append((name, para))

    # 权重衰减，通用的写法：bias和LayNorm.weight不用权重衰减，其他参数进行权重衰减。参见L2正则化
    # 此外还要差分学习率（differential learning），bert模型和其他模型学习率不同
    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        # bert模型
        {"params": [p for n, p in bert_param_optimizer if not any(nd in n for nd in no_decay)],
         "weight_decay": Config.weight_decay, 'lr': Config.bert_learning_rate},
        {"params": [p for n, p in bert_param_optimizer if any(nd in n for nd in no_decay)],
         "weight_decay": 0.0, 'lr': Config.bert_learning_rate},

        # 其他模型
        {"params": [p for n, p in other_param_optimizer if not any(nd in n for nd in no_decay)],
         "weight_decay": Config.weight_decay, 'lr': Config.other_learning_rate},
        {"params": [p for n, p in other_param_optimizer if any(nd in n for nd in no_decay)],
         "weight_decay": 0.0, 'lr':Config.other_learning_rate},
    ]

    # transformers的AdamW是目前最新的优化器，它和pytorch的AdamW参数格式统一
    optimizer = AdamW(optimizer_grouped_parameters, lr=Config.bert_learning_rate, eps=Config.adam_epsilon)

    # 学习率预热，训练时先从小的学习率开始训练 Learning Rate Scheduler，t_total是总步数
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=int(Config.warmup_proportion * t_total), num_training_steps=t_total
    )

    return optimizer, scheduler



if __name__ == '__main__':
 
    # 读取数据并进行预处理，load_data建议写成一个类（也就是以下内容都应该在load_data.py或者data_converter.py的类里以供这里调用）
    # 这里先拿2句话当作输入做测试，即batch_size=2，padding和mask这里先不做，但最终预处理的时候需要包含
    test_samples1 = '昨天清华着火了！'
    test_samples2 = '清华大学真厉害。'
    from transformers import BertTokenizer
    tokenizer = BertTokenizer.from_pretrained(Config.bert_dir)

    # input_ids就是一连串token在字典中的对应id。形状为(Config.batch_size, sequence_length)
    # sequence_length一般用padding把每句话的长度统一，长度在config.py里定义了，可以直接调
    input_ids=[]
    input_ids.append(tokenizer.encode(test_samples1))
    input_ids.append(tokenizer.encode(test_samples2))
    input_tensor=torch.LongTensor(input_ids).cuda()
    # print(input_tensor) 

    # attention_mask用于指定对哪些词进行self-Attention操作，
    # 避免在padding的token上计算attention（1不进行masked，0则masked）。形状为(Config.batch_size, sequence_length)。
    # 如果不设置则BertModel默认全为1
    # batch_mask = torch.ones_like(input_tensor) 




    #==================
    # 初始化模型
    trigger_extractor = TriggerExtractor() 
    trigger_extractor.to(Config.device)

    # 这里前向传播一次测试效果 这里应该再加一个attention_mask，目前为attention_mask=None
    output_tensor=trigger_extractor(input_tensor)
    # print(output_tensor.shape)
    print(output_tensor)

    #==================
    # 定义优化器
    t_total = 2 * Config.train_epochs # 目前以两句话为例子，最后训练用下面这个
    # t_total = len(train_loader) * opt.train_epochs
    optimizer, scheduler = build_optimizer_and_scheduler(trigger_extractor, t_total)
    
    #==================
    

