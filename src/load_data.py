import json
import random
import numpy as np
import torch
from transformers import BertTokenizer, BertForSequenceClassification, AdamW, BertModel



from config import Config

# 超参
SEED = 123
BATCH_SIZE = 16
LEARNING_RATE = 2e-5
WEIGHT_DECAY = 1e-2
EPSILON = 1e-8


# CorpusData 类，用来读入语料数据、存入语料数据
class CorpusData:


    # 按行读入 json，原数据集有误，每行json之间应该加逗号
    @staticmethod
    def load_json_file(file_path) -> list:
        f = open(file_path,'r')
        result=[]
        for line in f:
            result.append(json.loads(line))
        return result

    # 按 key 抽取 json 列表中的元素
    @staticmethod
    def json_list_extract_corpus_text(json_list, setence_key = 'text') -> list:
        setence = []
        for data in  json_list:
            setence.append(data[setence_key])
        return setence

    @staticmethod
    def json_list_extract_corpus_lable(json_list,label_key) -> list:
        labels = []
        for data in json_list:
            cur_label = []
            for label in data['labels']:
                cur_label.append(label[label_key])
            labels.append(cur_label)
        return labels


    #将每一句转成数字（大于126做截断，小于126做PADDING，加上首尾两个标识，长度总共等于128）
    @staticmethod
    def convert_text_to_token(tokenizer, sentence, limit_size=Config.sequence_length):

        tokens = tokenizer.encode(sentence[:limit_size])  #直接截断
        if len(tokens) < limit_size + 2:                  #补齐（pad的索引号就是0）
            tokens.extend([0] * (limit_size + 2 - len(tokens)))
        return tokens

    @staticmethod
    def convert_triggers_to_ndarray(triggers:list,sequence_length=Config.sequence_length)->np.ndarray:
        def convert_triggers_to_numpy(triggers,sequence_length):
            triggers_matrix = np.zeros([len(triggers),sequence_length],dtype=np.int8)
            sentence_index = 0
            for one_sentence_all_triggers in triggers:
                for trigger in one_sentence_all_triggers:
                    cur_index_start = trigger[1]
                    cur_index_end = trigger[1] + len(trigger[0])
                    triggers_matrix[sentence_index][cur_index_start:cur_index_end] = 1
                sentence_index += 1
            return triggers_matrix

        triggers_tensor = torch.from_numpy(convert_triggers_to_numpy(triggers,sequence_length))
        return triggers_tensor



        
    @staticmethod
    def convert_text_to_token_ndarray(tokenizer, all_sentences:list, limit_size = Config.sequence_length)->np.ndarray:
        sentences_size = len(all_sentences)
        all_sentences_tokens = np.zeros([sentences_size,limit_size],dtype=np.int32)

        for i in range(sentences_size):
            cur_token:list = tokenizer.encode(all_sentences[i][:(limit_size-2)])  #直接截断
            if len(cur_token) < limit_size:                  #补齐（pad的索引号就是0）
                cur_token.extend([0] * (limit_size - len(cur_token)))
            all_sentences_tokens[i] = cur_token
        return (all_sentences_tokens)



    @staticmethod
    #建立attention mask，PAD = 0，即input_all_sentences_ids 中的，非0值置为1
    def convert_tokens_to_attention_masks_ndarray(all_sentences_tokens:np.ndarray)->np.ndarray:
        #atten_masks_all_sentences = all_sentences_tokens.copy()
        atten_masks_all_sentences = np.where(all_sentences_tokens != 0, 1, 0)
        #atten_masks_all_sentences_tensor = torch.from_numpy(atten_masks_all_sentences)
        return atten_masks_all_sentences


        




def test_bert_model(sentences):
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)

    tokenizer = BertTokenizer.from_pretrained("hfl/chinese-roberta-wwm-ext", cache_dir="../saved_model/transformer_cached")
    model = BertModel.from_pretrained("hfl/chinese-roberta-wwm-ext")
    
    test_sentence = sentences[3]
    print(test_sentence)
    print(tokenizer.tokenize(test_sentence))
    print(tokenizer.encode(test_sentence))
    print(tokenizer.convert_ids_to_tokens(tokenizer.encode(test_sentence)))
    print(0)
    input_ids = [CorpusData.convert_text_to_token(tokenizer, sen) for sen in sentences]
    input_tokens = torch.tensor(input_ids)
    print(input_tokens.shape)                    #torch.Size([10000, 128])




if __name__ == "__main__":
    # execute only if run as a script
    
    dev_data_path = '../dataset/xf_2020_data/preliminary/raw_data/dev.json'
    dev_set = CorpusData.load_json_file(dev_data_path)
    sentences = CorpusData.json_list_extract_corpus_text(dev_set)
    #print(setence)

    
    triggers_word_and_id = CorpusData.json_list_extract_corpus_lable(dev_set,'trigger')
    print(triggers_word_and_id)

    trigger_tensor = CorpusData.convert_triggers_to_tensor(triggers_word_and_id)
    print(trigger_tensor)

    tokenizer = BertTokenizer.from_pretrained("hfl/chinese-roberta-wwm-ext", cache_dir="../saved_model/transformer_cached")
    model = BertModel.from_pretrained("hfl/chinese-roberta-wwm-ext")

    all_sentences_tokens = CorpusData.convert_text_to_token_ndarray(tokenizer,sentences);
    print(all_sentences_tokens[0])
    all_sentences_attention_masks = CorpusData.convert_tokens_to_attention_masks_ndarray(all_sentences_tokens);
    print(all_sentences_attention_masks[0])


#output
#     据日本共同社报道，日本东京都知事小池百合子在4月3日的记者会上，公布了中央政府基于“新冠病毒特措法”发布紧急事态宣言时的应对方针
# ['据', '日', '本', '共', '同', '社', '报', '道', '，', '日', '本', '东', '京', '都', '知', '事', '小', '池', '百', '合', '子', '在', '4', '月', '3', '日', '的', '记', '者', '会', '上', '，', '公', '布', '了', '中', '央', '政', '府', '基', '于', '[UNK]', '新', '冠', '病', '毒', '特', '措', '法', '[UNK]', '发', '布', '紧', '急', '事', '态', '宣', '言', '时', '的', '应', '对', '方', '针']
# [101, 2945, 3189, 3315, 1066, 1398, 4852, 2845, 6887, 8024, 3189, 3315, 691, 776, 6963, 4761, 752, 2207, 3737, 4636, 1394, 2094, 1762, 125, 3299, 124, 3189, 4638, 6381, 5442, 833, 677, 8024, 1062, 2357, 749, 704, 1925, 3124, 2424, 1825, 754, 100, 3173, 1094, 4567, 3681, 4294, 2974, 3791, 100, 1355, 2357, 5165, 2593, 752, 2578, 2146, 6241, 3198, 4638, 2418, 2190, 3175, 7151, 102]
# ['[CLS]', '据', '日', '本', '共', '同', '社', '报', '道', '，', '日', '本', '东', '京', '都', '知', '事', '小', '池', '百', '合', '子', '在', '4', '月', '3', '日', '的', '记', '者', '会', '上', '，', '公', '布', '了', '中', '央', '政', '府', '基', '于', '[UNK]', '新', '冠', '病', '毒', '特', '措', '法', '[UNK]', '发', '布', '紧', '急', '事', '态', '宣', '言', '时', '的', '应', '对', '方', '针', '[SEP]']
# 0