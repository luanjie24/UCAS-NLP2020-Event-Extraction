import os
import copy
import json
import torch
import logging
import numpy as np
from tqdm import tqdm
from config import Config
from transformers import BertTokenizer
from model import TriggerExtractor,SubObjExtractor,TimeLocExtractor
from processor import fine_grade_tokenize, search_label_index, ROLE2_TO_ID
from evaluator import pointer_trigger_decode2, pointer_decode, crf_decode
from convert_raw_data import clean_data

logger = logging.getLogger(__name__)
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO
)


def load_model_and_parallel(model, device, ckpt_path=None, strict=True):
    """
    加载模型 & 放置到 GPU 中
    """

    if ckpt_path is not None:
        logger.info(f'Load ckpt from {ckpt_path}')
        model.load_state_dict(torch.load(ckpt_path, map_location=torch.device('cpu')), strict=strict)
    if device != 'cup':
        model.to(device)

    # if len(gpu_ids) > 1:
    #     logger.info(f'Use multi gpus in: {gpu_ids}')
    #     gpu_ids = [int(x) for x in gpu_ids]
    #     model = torch.nn.DataParallel(model, device_ids=gpu_ids)
    # else:
    #     logger.info(f'Use single gpu in: {gpu_ids}')

    return model, device


def predict_stence(text):
    tokenizer = BertTokenizer.from_pretrained(Config.bert_dir)
    trigger_model = TriggerExtractor()
    sub_obj_model = SubObjExtractor()
    time_loc_model = TimeLocExtractor()
    trigger_model, device = load_model_and_parallel(trigger_model, 'cpu',
                                                    ckpt_path=os.path.join(Config.saved_best_trigger_extractor_dir, 'model.pt'))

    sub_obj_model, _ = load_model_and_parallel(sub_obj_model, 'cpu',
                                             ckpt_path=os.path.join(Config.saved_best_sub_obj_extractor_dir, 'model.pt'))
  
    time_loc_model, _ = load_model_and_parallel(time_loc_model, 'cup',
                                             ckpt_path=os.path.join(Config.saved_best_time_loc_extractor_dir, 'model.pt'))
    id2role = {ROLE2_TO_ID[key]: key for key in ROLE2_TO_ID.keys()}
    start_threshold = Config.start_threshold
    end_threshold = Config.end_threshold
    with torch.no_grad():
        #不适用droupout
        trigger_model.eval()
        sub_obj_model.eval()
        time_loc_model.eval()
        tmp_text_tokens = fine_grade_tokenize(text, tokenizer)

        assert len(text) == len(tmp_text_tokens)

        ## 无label标签
        trigger_encode_dict = tokenizer.encode_plus(text=tmp_text_tokens,
                                                    max_length=512,
                                                    pad_to_max_length=False,
                                                    is_pretokenized=True,
                                                    return_token_type_ids=True,
                                                    return_attention_mask=True,
                                                    return_tensors='pt')
        tmp_base_inputs = {'token_ids': trigger_encode_dict['input_ids'],
                            'attention_masks': trigger_encode_dict['attention_mask'],
                            'token_type_ids': trigger_encode_dict['token_type_ids']}

        trigger_inputs = copy.deepcopy(tmp_base_inputs)
        for key in trigger_inputs.keys():
            trigger_inputs[key] = trigger_inputs[key].to(device)
            

        #不需要label 啊哈哈哈哈啊哈哈
        tmp_trigger_pred = trigger_model.forward(trigger_inputs['token_ids'],trigger_inputs['attention_masks'],None)
        
        tmp_trigger_pred = tmp_trigger_pred[0][0].cpu().numpy()[1:1 + len(text)]
        
        tmp_triggers = pointer_trigger_decode2(tmp_trigger_pred, text, None,
                                                start_threshold=Config.start_threshold,
                                                end_threshold=Config.end_threshold,
                                                one_trigger=True)

        

        if not len(tmp_triggers):   #没有识别出触发词
            print(text)
            
        events = []  #保存事件

        for _trigger in tmp_triggers:
            tmp_event = {'trigger': {'text': _trigger[0],
                                        'length': len(_trigger[0]),
                                        'offset': int(_trigger[1])},
                            'arguments': []} 
                            
            if len(_trigger) > 2:
                print(_trigger)

            role_inputs = copy.deepcopy(tmp_base_inputs)
            trigger_start = _trigger[1] + 1
            trigger_end = trigger_start + len(_trigger[0]) - 1
            for i in range(trigger_start, trigger_end + 1):
                role_inputs['token_type_ids'][0][i] = 1

            tmp_trigger_label = torch.tensor([[trigger_start, trigger_end]]).long()

            role_inputs['trigger_index'] = tmp_trigger_label

            for key in role_inputs.keys():
                role_inputs[key] = role_inputs[key].to(device)
            
            tmp_roles_pred = sub_obj_model.forward(role_inputs['token_ids'],role_inputs['token_type_ids'],role_inputs['attention_masks'],None)
            # tmp_roles_pred = sub_obj_model(**role_inputs)[0][0].cpu().numpy()
            tmp_roles_pred = tmp_roles_pred[0][0].cpu().numpy()

            tmp_roles_pred = tmp_roles_pred[1:1 + len(text)]

            pred_obj = pointer_decode(tmp_roles_pred[:, :2], text, start_threshold, end_threshold, True)

            pred_sub = pointer_decode(tmp_roles_pred[:, 2:], text, start_threshold, end_threshold, True)

            if len(pred_obj) > 1:
                print(pred_obj)

            if len(pred_sub) > 1:
                print(pred_sub)

            pred_aux_tokens = time_loc_model.forward(role_inputs['token_ids'],role_inputs['token_type_ids'],role_inputs['attention_masks'],None)
            pred_aux = crf_decode(pred_aux_tokens[0], text, id2role)

            for _obj in pred_obj:
                tmp_event['arguments'].append({'role': 'object', 'text': _obj[0],
                                                'offset': int(_obj[1]), 'length': len(_obj[0])})
            for _sub in pred_sub:
                tmp_event['arguments'].append({'role': 'subject', 'text': _sub[0],
                                                'offset': int(_sub[1]), 'length': len(_sub[0])})

            for _role_type in pred_aux.keys():
                for _role in pred_aux[_role_type]:
                    tmp_event['arguments'].append({'role': _role_type, 'text': _role[0],
                                                    'offset': int(_role[1]), 'length': len(_role[0])})
            att_inputs = copy.deepcopy(tmp_base_inputs)

            att_inputs['trigger_index'] = tmp_trigger_label

            

            events.append(tmp_event)
        trigger = tmp_event['trigger']['text']
        subject = 'null'
        object_ = 'null'

        for item in tmp_event['arguments']:
            if item['role'] == 'subject':
                subject = item['text']
            if item['role'] == 'object':
                object_ = item['text']
        print(events)
        print('trigger:',trigger,'object:',object_,'subject:',subject)
        return trigger,subject,object_

if __name__ == "__main__":
    # pipeline_predict()
    predict_stence( "苏州市还成立合作区工作协调理事会，发挥高位协商作用，协调解决合作区改革发展面临的重大问题")