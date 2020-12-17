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


logger=logging.getLogger("train_info")
logger.setLevel(level=logging.DEBUG)
file_handler=handlers.TimedRotatingFileHandler(filename = Config.saved_log_dir, when = "H") 
file_handler.setLevel(level=logging.INFO)
stream_handler=logging.StreamHandler() 
stream_handler.setLevel(level=logging.DEBUG)
logger.addHandler(file_handler)
logger.addHandler(stream_handler)

def build_optimizer_and_scheduler(model, t_total):

    module = (
        model.module if hasattr(model, "module") else model
    )
    model_param = list(module.named_parameters())

    bert_param_optimizer = []
    other_param_optimizer = []
    for name, para in model_param:
        space = name.split('.')
        if space[0] == 'bert_module':
            bert_param_optimizer.append((name, para))
        else:
            other_param_optimizer.append((name, para))

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

    optimizer = AdamW(optimizer_grouped_parameters, lr=Config.bert_learning_rate, eps=Config.adam_epsilon)

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
    return os.path.join(output_dir, 'model.pt')

def save_best_model(model,save_model_dir):
    model_to_save = (
        model.module if hasattr(model, "module") else model
    )
    output_dir = os.path.join(save_model_dir, 'best_checkpoint')
    if not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
    torch.save(model_to_save.state_dict(), os.path.join(output_dir, 'model.pt'))
    print('save best checkpoint:\t',os.path.join(output_dir, 'model.pt'))

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
    train_sampler = RandomSampler(train_dataset)
    train_loader = DataLoader(dataset = train_dataset,
                              batch_size=Config.train_batch_size,
                              sampler=train_sampler,
                              num_workers=Config.num_workers)

    t_total = len(train_loader) * Config.train_epochs
    optimizer, scheduler = build_optimizer_and_scheduler(model, t_total)

    model.zero_grad()

    avg_acc = []
    avg_loss = 0. 
    f1 = 0. 
    max_F1 = 0.
    f1 = validate(model,num)
    global_step = 0
    log_loss_steps = 20
    save_steps = t_total // Config.train_epochs
    eval_steps = save_steps
    #==================
    # 训练
    logger.info("********** Running training **********")
    logger.info(f"  Num Examples = {len(train_dataset)}")
    logger.info(f"  Num Epochs = {Config.train_epochs}")
    logger.info(f"  Total training batch size = {Config.train_batch_size}")
    logger.info(f"  Total optimization steps = {t_total}")
    logger.info(f'Save model in {save_steps} steps; Eval model in {eval_steps} steps')
    logger.info(f'Save model at {save_model_dir}')


    for epoch in range(Config.train_epochs):
        is_best = False
        f1  = 0.
        for step, batch_data in enumerate(train_loader):

            for data in batch_data:
                data = data.to(Config.device)

            model.train()

            loss = model(*batch_data)[0] # 这么写是解压参数列表

            # 先将梯度归零，然后反向传播计算得到每个参数的梯度值，最后通过梯度下降执行一步参数更新
            model.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), Config.max_grad_norm)
            optimizer.step() 
            scheduler.step()

            global_step += 1
            
            # 每个batch的损失累加到avg_loss，每log_loss_steps算一次平均损失并输出，然后将损失清零
            if global_step % log_loss_steps == 0: 
                avg_loss /= log_loss_steps
                logger.info('Step: %d / %d ----> total loss: %.5f' % (global_step, t_total, avg_loss))
                avg_loss = 0.
            else:
                avg_loss += loss.item() 

        f1 = validate(model,num)
        if f1 > max_F1:
            is_best = True
            max_F1 = f1

        save_model(model, global_step, save_model_dir)
        if is_best:
            save_best_model(model,save_model_dir)

    # 释放显存
    torch.cuda.empty_cache()

    logger.info('Train done')
        



if __name__ == '__main__':

    # ===============预处理================
    corpus_data = CorpusData.load_corpus_data(Config.dataset_train)
    input_ids = torch.from_numpy(corpus_data["input_ids"]).long() 
    attention_masks = torch.from_numpy(corpus_data["attention_masks"]).long() 
    trigger_labels = torch.from_numpy(corpus_data["trigger_labels"]).long() 
    sub_obj_labels = torch.from_numpy(corpus_data["sub_obj_labels"]).long().cuda() 
    time_loc_labels = torch.from_numpy(corpus_data["time_loc_labels"]).long().cuda() 
    trigger_index = torch.from_numpy(corpus_data["trigger_index"]).long().cuda() 

    #==================trigger_extractor==================
    train_dataset = TensorDataset(input_ids, attention_masks, trigger_labels)
    trigger_extractor = TriggerExtractor() 
    trigger_extractor.to(Config.device)
    train(trigger_extractor, train_dataset, Config.saved_trigger_extractor_dir)
    
    #==================sub_obj_extractor==================
    train_dataset = TensorDataset(input_ids, trigger_index, attention_masks, sub_obj_labels)
    sub_obj_extractor = SubObjExtractor()
    sub_obj_extractor.to(Config.device)
    train(sub_obj_extractor, train_dataset, Config.saved_sub_obj_extractor_dir)

    #==================time_loc_extractor==================
    train_dataset = TensorDataset(input_ids, trigger_index, attention_masks, time_loc_labels)
    time_loc_extractor = TimeLocExtractor()
    time_loc_extractor.to(Config.device)
    train(time_loc_extractor, train_dataset, Config.saved_time_loc_extractor_dir)


    

    




