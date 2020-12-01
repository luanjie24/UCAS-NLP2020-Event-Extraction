import json

import numpy as np

from transformers import BertTokenizer, BertForSequenceClassification, AdamW, BertModel
from config import Config

# # 超参
# SEED = 123
# BATCH_SIZE = 16
# LEARNING_RATE = 2e-5
# WEIGHT_DECAY = 1e-2
# EPSILON = 1e-8


# CorpusData 类，用来读入语料数据、存入语料数据
class CorpusData:

    # 按行读入 json，原数据集有误，每行json之间应该加逗号
    @staticmethod
    def load_json_file(file_path) -> list:
        f = open(file_path, 'r')
        result = []
        for line in f:
            result.append(json.loads(line))
        return result

    # 按 key 抽取 json 列表中的元素
    @staticmethod
    def json_list_extract_corpus_text(json_list, setence_key='text') -> list:
        setence = []
        for data in json_list:
            setence.append(data[setence_key])
        return setence

    @staticmethod
    def json_list_extract_corpus_lable(json_list, label_key) -> list:
        labels = []
        for data in json_list:
            cur_label = []
            for label in data['labels']:
                cur_label.append(label[label_key])
            labels.append(cur_label)
        return labels

    # 将每一句转成数字（大于126做截断，小于126做PADDING，加上首尾两个标识，长度总共等于128）
    # 参考 Transformer 的 encode_plus 文档
    # https://huggingface.co/transformers/internal/tokenization_utils.html#transformers.tokenization_utils_base.PreTrainedTokenizerBase.encode_plus
    @staticmethod
    def convert_text_to_token_ndarray(tokenizer, all_sentences: list, limit_size=Config.sequence_length) -> (np.ndarray, np.ndarray):
        sentences_size = len(all_sentences)
        all_sentences_tokens = np.zeros(
            [sentences_size, limit_size], dtype=np.int32)
        all_sentences_attention_masks = np.zeros(
            [sentences_size, limit_size], dtype=np.int32)

        for i in range(sentences_size):
            cur_token = tokenizer.encode_plus(
                all_sentences[i], padding='max_length', truncation=True, max_length=Config.sequence_length)
            all_sentences_tokens[i] = cur_token['input_ids']
            all_sentences_attention_masks[i] = cur_token['attention_mask']

        return all_sentences_tokens, all_sentences_attention_masks

    # 把所有触发词，映射到tokens，目的是忽略坐标
    @staticmethod
    def convert_labels_to_tokens(tokenizer, labels: list) -> list:
        all_sentences_lables = []
        # one_sentence_labels example :[['找到', 21], ['接受', 63]]
        for one_sentence_labels in labels:
            cur_sentence_labels = []
            # one label example : ['找到', 21] is a LIST!!!!!
            for one_label in one_sentence_labels:
                if one_label:  # 标签有可能为空
                    one_label = tokenizer.encode(
                        one_label[0], add_special_tokens=False)
                    cur_sentence_labels.append(one_label)
            all_sentences_lables.append(cur_sentence_labels)
        return all_sentences_lables

    # 把所有触发词的位置在 encode 之后的句子中定位到，并打标签
    # https://stackoverflow.com/questions/53352767/finding-the-index-of-a-sublist-in-a-numpy-array
    @staticmethod
    def convert_labels_to_mask_ndarray(tokenizer, all_sentences_labels: list, all_sentences_tokens: np.ndarray) -> np.ndarray:
        def rolling_window(a, size):
            shape = a.shape[:-1] + (a.shape[-1] - size + 1, size)
            strides = a.strides + (a. strides[-1],)
            return np.lib.stride_tricks.as_strided(a, shape=shape, strides=strides)

        all_lables_tokens: list = CorpusData.convert_labels_to_tokens(
            tokenizer, all_sentences_labels)

        # 创建 标签掩码矩阵
        all_lables_mask: np.ndarray = np.zeros(
            all_sentences_tokens.shape, dtype=np.int8)
        # 根据lables 在  all_sentences_tokens 中进行匹配
        sentences_size = len(all_lables_tokens)
        for idx in range(sentences_size):
            cur_sentence_lables = all_lables_tokens[idx]
            for cur_lable in cur_sentence_lables:
                cur_label_len = len(cur_lable)
                cur_label_ndarray = np.array(cur_lable)
                label_is_exist = np.all(rolling_window(
                    all_sentences_tokens[idx], cur_label_len) == cur_label_ndarray, axis=1)
                label_index_tuple = np.where(label_is_exist == True)
                label_index = label_index_tuple[0][0]
                all_lables_mask[idx][label_index:label_index+cur_label_len] = 1

        return all_lables_mask
        # return triggers_tensor

    # 外部调用这个函数！！！
    # 读取数据的完整函数！！！！返回一个 dict, {'input_ids': 输入词对应token id，'attention_masks': }
    @staticmethod
    def load_corpus_data(file_name: str = '', load_data_from_cache: bool = False, cached_file_name: str = '') -> dict:

        # 从数据文件中直接读取多个数组
        if load_data_from_cache:
            cached_file_name = cached_file_name+'.npy'
            corpus_data = np.load(cached_file_name, allow_pickle=True)
            print("数据加载完毕")
            return corpus_data
        # 从Json FILE中读取数组
        else:
            data_set_json = CorpusData.load_json_file(dev_data_path)
            # 读入所有语料中的句子
            sentences = CorpusData.json_list_extract_corpus_text(data_set_json)

            tokenizer = BertTokenizer.from_pretrained(
                "hfl/chinese-roberta-wwm-ext", cache_dir="../saved_model/transformer_cached")

            # 将句子转换为 token，同时创建Attention mask
            all_sentences_tokens, all_sentences_attention_masks = CorpusData.convert_text_to_token_ndarray(
                tokenizer, sentences)

            # 以下生成标签

            # 读入所有触发词，每句话可能有多个触发词
            triggers_word_and_index = CorpusData.json_list_extract_corpus_lable(
                data_set_json, 'trigger')

            # 创建触发词标签mask
            all_sentences_triggers_labels = CorpusData.convert_labels_to_mask_ndarray(
                tokenizer, triggers_word_and_index, all_sentences_tokens)

            # 读入所有 Object
            # 读入所有Object 主语，每句话可能有多个触发词
            object_word_and_index = CorpusData.json_list_extract_corpus_lable(
                data_set_json, 'object')

            # 创建主语标签mask
            all_sentences_objects_labels = CorpusData.convert_labels_to_mask_ndarray(
                tokenizer, object_word_and_index, all_sentences_tokens)

            # 读入所有 Subject
            subject_word_and_index = CorpusData.json_list_extract_corpus_lable(
                data_set_json, 'subject')
            # 创建Subject 宾语标签mask
            all_sentences_subjects_labels = CorpusData.convert_labels_to_mask_ndarray(
                tokenizer, subject_word_and_index, all_sentences_tokens)

            # 最后构造一个 dict
            corpus_data = {'input_ids': all_sentences_tokens, 'attention_masks': all_sentences_attention_masks,
                           'trigger_labels': all_sentences_triggers_labels, 'object_labels': all_sentences_objects_labels,
                           'subject_labels': all_sentences_subjects_labels}
            return corpus_data

    # https://numpy.org/doc/stable/reference/generated/numpy.load.html
    # 必须指明 allow_pickle = True
    # 保存 dict of numpy.ndarray 
    @staticmethod
    def save_corpus_data(all_corpus_data: dict, output_file_name: str):
        np.save(output_file_name, all_corpus_data, allow_pickle=True)
        print("数据储存完毕")
        return


def test_bert_model(sentences):

    dev_data_path = '../dataset/xf_2020_data/preliminary/raw_data/dev.json'
    dev_set = CorpusData.load_json_file(dev_data_path)
    sentences = CorpusData.json_list_extract_corpus_text(dev_set)
    print(sentences[0])

    triggers_word_and_id = CorpusData.json_list_extract_corpus_lable(
        dev_set, 'trigger')
    # print(triggers_word_and_id)

    # trigger_tensor = CorpusData.convert_triggers_to_tensor(triggers_word_and_id)
    # print(trigger_tensor)

    tokenizer = BertTokenizer.from_pretrained(
        "hfl/chinese-roberta-wwm-ext", cache_dir="../saved_model/transformer_cached")
    model = BertModel.from_pretrained("hfl/chinese-roberta-wwm-ext")

    all_sentences_tokens, all_sentences_attention_masks = CorpusData.convert_text_to_token_ndarray(
        tokenizer, sentences)
    print(all_sentences_tokens[0])
    # all_sentences_attention_masks = CorpusData.convert_tokens_to_attention_masks_ndarray(all_sentences_tokens);
    print(all_sentences_attention_masks[0])

    all_sentences_labels_token = CorpusData.convert_labels_to_tokens(
        tokenizer, triggers_word_and_id)
    print(all_sentences_labels_token[0])

    all_sentences_triggers_mask = CorpusData.convert_labels_to_mask_ndarray(
        tokenizer, triggers_word_and_id, all_sentences_tokens)
    print(all_sentences_triggers_mask[0])

    np.save("test.npy", all_sentences_attention_masks)
# output
#     据日本共同社报道，日本东京都知事小池百合子在4月3日的记者会上，公布了中央政府基于“新冠病毒特措法”发布紧急事态宣言时的应对方针
# ['据', '日', '本', '共', '同', '社', '报', '道', '，', '日', '本', '东', '京', '都', '知', '事', '小', '池', '百', '合', '子', '在', '4', '月', '3', '日', '的', '记', '者', '会', '上', '，', '公', '布', '了', '中', '央', '政', '府', '基', '于', '[UNK]', '新', '冠', '病', '毒', '特', '措', '法', '[UNK]', '发', '布', '紧', '急', '事', '态', '宣', '言', '时', '的', '应', '对', '方', '针']
# [101, 2945, 3189, 3315, 1066, 1398, 4852, 2845, 6887, 8024, 3189, 3315, 691, 776, 6963, 4761, 752, 2207, 3737, 4636, 1394, 2094, 1762, 125, 3299, 124, 3189, 4638, 6381, 5442, 833, 677, 8024, 1062, 2357, 749, 704, 1925, 3124, 2424, 1825, 754, 100, 3173, 1094, 4567, 3681, 4294, 2974, 3791, 100, 1355, 2357, 5165, 2593, 752, 2578, 2146, 6241, 3198, 4638, 2418, 2190, 3175, 7151, 102]
# ['[CLS]', '据', '日', '本', '共', '同', '社', '报', '道', '，', '日', '本', '东', '京', '都', '知', '事', '小', '池', '百', '合', '子', '在', '4', '月', '3', '日', '的', '记', '者', '会', '上', '，', '公', '布', '了', '中', '央', '政', '府', '基', '于', '[UNK]', '新', '冠', '病', '毒', '特', '措', '法', '[UNK]', '发', '布', '紧', '急', '事', '态', '宣', '言', '时', '的', '应', '对', '方', '针', '[SEP]']
# 0


if __name__ == "__main__":

    # 这里的 main 函数仅作测试用，具体参数变量参考 Config 文件
    dev_data_path = '../dataset/xf_2020_data/preliminary/raw_data/dev.json'
    corpus_data = CorpusData.load_corpus_data(dev_data_path)

    # save data
    CorpusData.save_corpus_data(corpus_data, '../saved_model/dev_data')

    # 加载数据的时候，默认使用 .npy 后缀名！
    corpus_data_from_cached = CorpusData.load_corpus_data(
        load_data_from_cache=True, cached_file_name='../saved_model/dev_data')


#     {'input_ids': array([[ 101,  100,  125, ...,    0,    0,    0],
#         [ 101, 1762, 3634, ...,    0,    0,    0],
#         [ 101, 2190,  754, ...,    0,    0,    0],
#         ...,
#         [ 101, 9093, 2399, ...,    0,    0,    0],
#         [ 101, 1094, 4495, ...,    0,    0,    0],
#         [ 101,  100, 8020, ...,    0,    0,    0]], dtype=int32),
#  'attention_masks': array([[1, 1, 1, ..., 0, 0, 0],
#         [1, 1, 1, ..., 0, 0, 0],
#         [1, 1, 1, ..., 0, 0, 0],
#         ...,
#         [1, 1, 1, ..., 0, 0, 0],
#         [1, 1, 1, ..., 0, 0, 0],
#         [1, 1, 1, ..., 0, 0, 0]], dtype=int32),
#  'trigger_lables': array([[0, 0, 0, ..., 0, 0, 0],
#         [0, 0, 0, ..., 0, 0, 0],
#         [0, 0, 0, ..., 0, 0, 0],
#         ...,
#         [0, 0, 0, ..., 0, 0, 0],
#         [0, 0, 0, ..., 0, 0, 0],
#         [0, 0, 0, ..., 0, 0, 0]], dtype=int8)}
