"""
train.py用于训练集的训练

"""

import os
import logging
from logging import handlers
import torch
from torch.utils.data import DataLoader, RandomSampler, TensorDataset
from config import Config
from model import TriggerExtractor
from transformers import AdamW, get_linear_schedule_with_warmup
from load_data import CorpusData


# 将训练日志输出到控制台和文件
logger=logging.getLogger("train_info")
logger.setLevel(level=logging.DEBUG)
file_handler=handlers.TimedRotatingFileHandler(filename = Config.saved_log_dir, when = "H") # 输出周期日志文件
file_handler.setLevel(level=logging.INFO)
stream_handler=logging.StreamHandler() # 输出到控制台
stream_handler.setLevel(level=logging.DEBUG)
logger.addHandler(file_handler)
logger.addHandler(stream_handler)

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


def save_model(model, global_step, saved_dir):
    output_dir = os.path.join(saved_dir, 'checkpoint-{}'.format(global_step))
    if not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)

    # take care of model distributed / parallel training
    model_to_save = (
        model.module if hasattr(model, "module") else model
    )
    logger.info(f'Saving model & optimizer & scheduler checkpoint to {output_dir}')
    torch.save(model_to_save.state_dict(), os.path.join(output_dir, 'model.pt'))



def train(model, train_dataset):
    # 这个train_dataset目前是传样本和label，也可以改成只传样本
    train_sampler = RandomSampler(train_dataset) # 打乱顺序，sampler为取样本的策略，功能应该和shuffle差不多
    train_loader = DataLoader(dataset=train_dataset,
                              batch_size=Config.train_batch_size,
                              sampler=train_sampler,
                              num_workers=0) # 这块开线程会报错

    # 测试，查看一下train_dataloader的内容，之后有了mask就变成or i, (train, mask,label) in enumerate(train_loader)
    # for i, (train, train_mask, label) in enumerate(train_loader):
    #     print(train.shape, label.shape)
    #     print(train) # 对于trigger_extractor，输出的是一个tensor shape(Config.train_batch_size，sequence_length)
    #     print(train_mask) # 对于trigger_extractor，输出的是一个tensor shape(Config.train_batch_size，sequence_length)
    #     print(label) # 对于trigger_extractor，输出的是一个tensor shape(Config.train_batch_size，sequence_length，2) 2是每个字两个值
    #     break
    # print('len(train_loader)=', len(train_loader))
    # for step, batch_data in enumerate(train_loader):
    #     print("step:",step)
    #     print("batch_data:",batch_data)

    #==================
    # 定义优化器
    t_total = len(train_loader) * Config.train_epochs # 定义总的更新轮数，len(train_loader)我觉得就是每个epoch更新学习率的次数
    optimizer, scheduler = build_optimizer_and_scheduler(trigger_extractor, t_total)

    model.zero_grad()

    avg_acc = [] # 准确率，没实现
    avg_loss = 0. # 平均损失
    global_step = 0 # 总训练步数
    log_loss_steps = 20 # 每log_loss_steps步算一次平均损失
    save_steps = t_total // Config.train_epochs # 每save_steps存一次模型，没实现
    eval_steps = save_steps # 每eval_steps算一次准确率，没实现
    #==================
    # 训练
    logger.info("********** Running training **********")
    logger.info(f"  Num Examples = {len(train_dataset)}")
    logger.info(f"  Num Epochs = {Config.train_epochs}")
    logger.info(f"  Total training batch size = {Config.train_batch_size}")
    logger.info(f"  Total optimization steps = {t_total}")
    logger.info(f'Save model in {save_steps} steps; Eval model in {eval_steps} steps')
    
    for epoch in range(Config.train_epochs):
        # 之后有了mask就变成for i, (train, mask,label) in enumerate(train_loader)
        for step, (batch_train_data, batch_train_mask, batch_train_label) in enumerate(train_loader):
            
            # 数据运算迁移到GPU
            batch_train_data, batch_train_mask, batch_train_label = batch_train_data.to(Config.device), batch_train_mask.to(Config.device), batch_train_label.to(Config.device)

            # 改成batch_size个句子都训练
            model.train()

            output_tensor=model(batch_train_data, attention_mask = batch_train_mask, labels = batch_train_label)

            # 如果是trigger_extractor输出是这个格式，其他模型还得看看能不能设计成这个格式
            loss, logits = output_tensor[0], output_tensor[1]
            
            # 每个batch更新权重的一些必要操作
            # 总得来说，先将梯度归零，然后反向传播计算得到每个参数的梯度值，最后通过梯度下降执行一步参数更新
            model.zero_grad() # 梯度清零
            loss.backward() # 反向传播，计算当前梯度；
            torch.nn.utils.clip_grad_norm_(model.parameters(), Config.max_grad_norm) # 梯度不能大于Config.max_grad_norm，防止梯度爆炸 注意这个方法只在训练的时候使用，在测试的时候不用
            optimizer.step() # 更新模型参数
            scheduler.step() # 更新learning rate

            global_step += 1
            
            # 每个batch的损失累加到avg_loss，每log_loss_steps算一次平均损失并输出，然后将损失清零
            if global_step % log_loss_steps == 0: 
                avg_loss /= log_loss_steps
                logger.info('Step: %d / %d ----> total loss: %.5f' % (global_step, t_total, avg_loss))
                avg_loss = 0.
            else:
                avg_loss += loss.item() 

            # 每save_steps存一次模型
            if global_step % save_steps == 0:
                save_model(model, global_step, Config.saved_trigger_extractor_dir)

    # 释放显存
    torch.cuda.empty_cache()

    logger.info('Train done')
        



if __name__ == '__main__':
 
    '''
    # 读取数据并进行预处理的测试（等价于load_data里的操作）
    # 这里先拿2句话当作输入做测试，即batch_size=2，padding和mask这里先不做，但最终预处理的时候需要包含
    test_samples1 = '昨天清华着火了！'
    test_samples2 = '清华大学真厉害。'

    # 触发词识别标签的测试样例               
    samples_y = torch.tensor([[[0,0],[0,0],[0,0],[0,0],[0,0],[1,0],[0,1],[0,0],[0,0],[0,0]],
                    [[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0]]]).cuda()
    from transformers import BertTokenizer
    tokenizer = BertTokenizer.from_pretrained(Config.bert_dir)

    # input_ids就是一连串token在字典中的对应id。形状为(Config.batch_size, sequence_length)
    # sequence_length一般用padding把每句话的长度统一，长度在config.py里定义了，可以直接调
    input_ids=[]
    input_ids.append(tokenizer.encode(test_samples1))
    input_ids.append(tokenizer.encode(test_samples2))
    input_tensor=torch.LongTensor(input_ids).cuda()
    # print(input_tensor.shape)
    # print(samples_y.shape) 

    '''
    #==================
    # 拿到训练数据集
    corpus_data = CorpusData.load_corpus_data(Config.dataset_train)
    # numpy转tensor
    input_tensor = torch.from_numpy(corpus_data["input_ids"]).long() 
    train_masks = torch.from_numpy(corpus_data["attention_masks"]).long() 
    samples_y = torch.from_numpy(corpus_data["trigger_labels"]).long() 
    # 对于触发词识别，打包数据集形成DataLoader需求的dataset去
    train_dataset = TensorDataset(input_tensor, train_masks, samples_y)

    #==================
    # 初始化模型
    trigger_extractor = TriggerExtractor() 
    trigger_extractor.to(Config.device)

    # 测试，前向传播一次
    # output_tensor=trigger_extractor(input_tensor, train_masks, labels=test_labels[0])
    # # print(output_tensor.shape)
    # print(output_tensor) # 如果有label，返回的是两个元素的元组，第一个元素是损失的张量，第二个元素是前向传播的y

    train(trigger_extractor, train_dataset)




    

    




