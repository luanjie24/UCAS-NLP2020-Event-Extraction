"""
train.py用于训练集的训练

"""

import os
import logging
from logging import handlers
import torch
from torch.utils.data import DataLoader, RandomSampler, TensorDataset
from transformers import AdamW, get_linear_schedule_with_warmup
from config import Config
from model import TriggerExtractor, SubObjExtractor, TimeLocExtractor
from load_data import CorpusData

from evaluator import trigger_evaluation, role1_evaluation, role2_evaluation, attribution_evaluation
from torch.autograd import Variable

from processor import *
from dataset_utils import build_dataset

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


def save_model(model, global_step, saved_dir,is_best):
    output_dir = os.path.join(saved_dir, 'checkpoint-{}'.format(global_step))
    if not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
    
    # take care of model distributed / parallel training
    model_to_save = (
        model.module if hasattr(model, "module") else model
    )
    logger.info(f'Saving model & optimizer & scheduler checkpoint to {output_dir}')
    torch.save(model_to_save.state_dict(), os.path.join(output_dir, 'model.pt'))

    if is_best:
        output_best_dir = os.path.join(saved_dir, 'best_checkpoint-{}'.format(global_step))
        if not os.path.exists(output_best_dir):
            os.makedirs(output_best_dir, exist_ok=True)
        torch.save(model_to_save.state_dict(), os.path.join(output_best_dir, 'model.pt'))


def validate(model,num):
    processors = {1: TriggerProcessor,
                  2: RoleProcessor,
                  3: RoleProcessor}

    processor = processors[num]()
    dev_raw_examples = processor.read_json(Config.dataset_dev)
    dev_examples, dev_callback_info = processor.get_dev_examples(dev_raw_examples)
    dev_features = convert_examples_to_features(num, dev_examples, Config.bert_dir,
                                                Config.sequence_length)
    
    dev_dataset = build_dataset(num, dev_features,
                                mode='dev')
    dev_loader = DataLoader(dev_dataset, batch_size=Config.dev_batch_size,
                            shuffle=False, num_workers=8)
    dev_info = (dev_loader, dev_callback_info)

    # model = build_model(opt.task_type, opt.bert_dir, **model_para)
    if num == 1:
        tmp_metric_str, tmp_f1 = trigger_evaluation(model, dev_info, Config.device,
                                                        start_threshold=Config.start_threshold,
                                                        end_threshold=Config.end_threshold)
    elif num == 2:
        tmp_metric_str, tmp_f1 = role1_evaluation(model, dev_info, Config.device,
                                                        start_threshold=Config.start_threshold,
                                                        end_threshold=Config.end_threshold)
    elif num == 3:
        tmp_metric_str, tmp_f1 = role2_evaluation(model, dev_info, Config.device)
    else:
        print('error model')
    print('tmp_f1:\t',tmp_f1)
    return tmp_f1

def train(model, train_dataset, save_model_dir,num):
    train_sampler = RandomSampler(train_dataset) # 打乱顺序，sampler为取样本的策略，功能应该和shuffle差不多
    train_loader = DataLoader(dataset = train_dataset,
                              batch_size=Config.train_batch_size,
                              sampler=train_sampler,
                              num_workers=Config.num_workers)

    # 测试，查看一下train_dataloader的内容
    # print('len(train_loader)=', len(train_loader))
    # for step, batch_data in enumerate(train_loader):
    #     print("step:",step)
    #     print("batch_data:",batch_data)
    #==================
    # 定义优化器
    t_total = len(train_loader) * Config.train_epochs # 定义总的更新轮数，len(train_loader)就是每个epoch更新学习率的次数
    optimizer, scheduler = build_optimizer_and_scheduler(model, t_total)

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
    logger.info(f'Save model at {save_model_dir}')
    f1 = 0.
    max_F1 = 0.
    for epoch in range(Config.train_epochs):
        is_best = False
        f1  = 0.
        for step, batch_data in enumerate(train_loader):

            # 数据运算迁移到GPU
            for data in batch_data:
                data = data.to(Config.device)

            model.train()

            loss = model(*batch_data)[0] # 这么写是解压参数列表

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
            # if global_step % save_steps == 0:
            #     save_model(model, global_step, save_model_dir)

        f1 = validate(model,num)
        if f1 > max_F1:
            is_best = True
            max_F1 = f1
        
        save_model(model, global_step, save_model_dir,is_best)

    # 释放显存
    torch.cuda.empty_cache()

    logger.info('Train done')
        



if __name__ == '__main__':

    '''
    # ======================预处理数据生成（仅测试用）=======================
    # 读取数据并进行预处理的测试（等价于load_data里的操作）
    # 这里先拿2句话当作输入做测试，即batch_size=2
    test_samples1 = '昨天清华着火了！'
    test_samples2 = '杰伦参加演唱会。'

    from transformers import BertTokenizer
    tokenizer = BertTokenizer.from_pretrained(Config.bert_dir)

    # input_ids就是一连串token在字典中的对应id。shape：(Config.batch_size, Config.sequence_length)
    input_ids=[]
    input_ids.append(tokenizer.encode(test_samples1))
    input_ids.append(tokenizer.encode(test_samples2))
    input_ids=torch.LongTensor(input_ids).cuda()

    # attention_masks是对padding部分进行mask，有字部分为1，padding部分为0。shape:(Config.batch_size, Config.sequence_length)
    attention_masks=torch.LongTensor([[1,1,1,1,1,1,1,1,1,1],
                    [1,1,1,1,1,1,1,1,1,1]]).cuda()

    # 触发词提取模型测试样例的label。shape：(Config.batch_size, Config.sequence_length, 2)             
    trigger_labels = torch.tensor([[[0,0],[0,0],[0,0],[0,0],[0,0],[1,0],[0,1],[0,0],[0,0],[0,0]],
                    [[0,0],[0,0],[0,0],[1,0],[0,1],[0,0],[0,0],[0,0],[0,0],[0,0]]]).cuda()

    # 主客体识别模型测试样例的label。shape:（Config.train_batch_size, Config.sequence_length, 4）
    sub_obj_labels = torch.tensor([[[0,0,0,0],[0,0,0,0],[0,0,0,0],[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1],[0,0,0,0],[0,0,0,0],[0,0,0,0]],
                    [[0,0,0,0],[1,0,0,0],[0,1,0,0],[0,0,0,0],[0,0,0,0],[0,0,1,0],[0,0,0,0],[0,0,0,1],[0,0,0,0],[0,0,0,0]]]).cuda()

    # 主体客体识别模型测试样例的trigger_index
    # trigger_index shape:(Config.train_batch_size, n) trigger_index应该是trigger第一个字和最后一个字在文本中的位置（n应该永远等于2）
    trigger_index=torch.LongTensor([[5, 6],[3,4]]).cuda()

    # 主体客体识别模型测试样例的trigger_distance，这个先不做了
    # trigger_distance=torch.LongTensor([[5,4,3,2,1,0,0,1,2,3],[3,2,1,0,0,1,2,3,4,5]]).cuda()

    # 时间地点识别模型测试样例的label。shape:（Config.train_batch_size, Config.sequence_length）
    # (B-begin，I-inside，O-outside，E-end，S-single)
    # TIME_LOC_TO_ID = {
    # "O": 0,
    # "B-time": 1,
    # "I-time": 2,
    # "E-time": 3,
    # "S-time": 4,
    # "B-loc": 5,
    # "I-loc": 6,
    # "E-loc": 7,
    # "S-loc": 8,
    # "X": 9 # 这个好像是因为要随机丢弃70%负样本才定义的，暂时不考虑了吧
    # }
    time_loc_labels = torch.tensor([[0,1,2,5,7,0,0,0,0,0],
                    [0,0,0,0,0,0,0,0,0,0]]).cuda()
    
    '''

    # ===============拿到预处理后的数据并转tensor================
    corpus_data = CorpusData.load_corpus_data(Config.dataset_train)
    input_ids = torch.from_numpy(corpus_data["input_ids"]).long() 
    attention_masks = torch.from_numpy(corpus_data["attention_masks"]).long() 
    trigger_labels = torch.from_numpy(corpus_data["trigger_labels"]).long() 
    sub_obj_labels = torch.from_numpy(corpus_data["sub_obj_labels"]).long().cuda() 
    time_loc_labels = torch.from_numpy(corpus_data["time_loc_labels"]).long().cuda() 
    trigger_index = torch.from_numpy(corpus_data["trigger_index"]).long().cuda() 


     
    #==================trigger_extractor==================
    # 数据集打包
    train_dataset = TensorDataset(input_ids, attention_masks, trigger_labels)
    # 初始化模型
    trigger_extractor = TriggerExtractor() 
    trigger_extractor.to(Config.device)
    # 测试，前向传播一次
    # output_tensor=trigger_extractor(input_ids, attention_masks, labels=trigger_labels)
    # print(output_tensor)
    # 训练
    train(trigger_extractor, train_dataset, Config.saved_trigger_extractor_dir)
    

    '''
    #==================sub_obj_extractor==================
    # 数据集打包
    train_dataset = TensorDataset(input_ids, trigger_index, attention_masks, sub_obj_labels)

    # 初始化模型
    sub_obj_extractor = SubObjExtractor()
    sub_obj_extractor.to(Config.device)

    # 测试，前向传播一次
    # output=sub_obj_extractor(input_ids, trigger_index, attention_masks, labels = sub_obj_labels)
    # print(output)

    train(sub_obj_extractor, train_dataset, Config.saved_sub_obj_extractor_dir)
    '''

    '''
    #==================time_loc_extractor==================
    # 数据集打包
    train_dataset = TensorDataset(input_ids, trigger_index, attention_masks, time_loc_labels)

    # 初始化模型
    time_loc_extractor = TimeLocExtractor()
    time_loc_extractor.to(Config.device)

    # 测试，前向传播一次
    # output = time_loc_extractor(input_ids, trigger_index, attention_masks, labels = time_loc_labels)
    # print(output)

    train(time_loc_extractor, train_dataset, Config.saved_time_loc_extractor_dir)
    '''

    

    




