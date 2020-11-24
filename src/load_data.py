import json
import random
import numpy as np
import torch
from transformers import BertTokenizer, BertForSequenceClassification, AdamW, BertModel


# 超参
SEED = 123
BATCH_SIZE = 16
LEARNING_RATE = 2e-5
WEIGHT_DECAY = 1e-2
EPSILON = 1e-8


# 按行读入 json，原数据集有误，每行json之间应该加逗号
def load_json_file(file_path) -> list:
    f = open(file_path,'r')
    result=[]
    for line in f:
        result.append(json.loads(line))
    return result

# 按 key 抽取 json 列表中的元素
def json_list_extract_data(json_list, setence_key) -> list:
    setence = []
    for data in  json_list:
        setence.append(data[setence_key])
    return setence

#将每一句转成数字（大于126做截断，小于126做PADDING，加上首尾两个标识，长度总共等于128）
def convert_text_to_token(tokenizer, sentence, limit_size=126):

    tokens = tokenizer.encode(sentence[:limit_size])  #直接截断
    if len(tokens) < limit_size + 2:                  #补齐（pad的索引号就是0）
        tokens.extend([0] * (limit_size + 2 - len(tokens)))
    return tokens


if __name__ == "__main__":
    # execute only if run as a script
    dev_data_path = '../dataset/xf_2020_data/preliminary/raw_data/dev.json'
    dev_set = load_json_file(dev_data_path)
    sentences = json_list_extract_data(dev_set,'text')
    #print(setence)

    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)

    tokenizer = BertTokenizer.from_pretrained("hfl/chinese-roberta-wwm-ext", cache_dir="../saved_model/transformer_cached")
    model = BertModel.from_pretrained("hfl/chinese-roberta-wwm-ext")
    
    test_sentence = sentences[5]
    print(test_sentence)
    print(tokenizer.tokenize(test_sentence))
    print(tokenizer.encode(test_sentence))
    print(tokenizer.convert_ids_to_tokens(tokenizer.encode(test_sentence)))
    print(0)
    input_ids = [convert_text_to_token(tokenizer, sen) for sen in sentences]
    input_tokens = torch.tensor(input_ids)
    print(input_tokens.shape)                    #torch.Size([10000, 128])

#output
#     据日本共同社报道，日本东京都知事小池百合子在4月3日的记者会上，公布了中央政府基于“新冠病毒特措法”发布紧急事态宣言时的应对方针
# ['据', '日', '本', '共', '同', '社', '报', '道', '，', '日', '本', '东', '京', '都', '知', '事', '小', '池', '百', '合', '子', '在', '4', '月', '3', '日', '的', '记', '者', '会', '上', '，', '公', '布', '了', '中', '央', '政', '府', '基', '于', '[UNK]', '新', '冠', '病', '毒', '特', '措', '法', '[UNK]', '发', '布', '紧', '急', '事', '态', '宣', '言', '时', '的', '应', '对', '方', '针']
# [101, 2945, 3189, 3315, 1066, 1398, 4852, 2845, 6887, 8024, 3189, 3315, 691, 776, 6963, 4761, 752, 2207, 3737, 4636, 1394, 2094, 1762, 125, 3299, 124, 3189, 4638, 6381, 5442, 833, 677, 8024, 1062, 2357, 749, 704, 1925, 3124, 2424, 1825, 754, 100, 3173, 1094, 4567, 3681, 4294, 2974, 3791, 100, 1355, 2357, 5165, 2593, 752, 2578, 2146, 6241, 3198, 4638, 2418, 2190, 3175, 7151, 102]
# ['[CLS]', '据', '日', '本', '共', '同', '社', '报', '道', '，', '日', '本', '东', '京', '都', '知', '事', '小', '池', '百', '合', '子', '在', '4', '月', '3', '日', '的', '记', '者', '会', '上', '，', '公', '布', '了', '中', '央', '政', '府', '基', '于', '[UNK]', '新', '冠', '病', '毒', '特', '措', '法', '[UNK]', '发', '布', '紧', '急', '事', '态', '宣', '言', '时', '的', '应', '对', '方', '针', '[SEP]']
# 0