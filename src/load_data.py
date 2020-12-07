import json

import numpy as np
from typing import List
from transformers import BertTokenizer, BertForSequenceClassification, AdamW, BertModel
from config import Config


# # 超参
# SEED = 123
# BATCH_SIZE = 16
# LEARNING_RATE = 2e-5
# WEIGHT_DECAY = 1e-2
# EPSILON = 1e-8


class Event:
    m_sentence: str
    m_trigger: str
    m_object: str
    m_subject: str
    m_time: str
    m_location: str

    # 一条文本的最大长度
    m_max_length: int
    # 每个单词的Token
    # 如果没有某个论元，则其 token 为 np.zeros((1, 1))
    m_trigger_token: np.ndarray
    m_object_token: np.ndarray
    m_subject_token: np.ndarray
    m_time_token: np.ndarray
    m_location_token: np.ndarray

    # 输入，一句话的Token + 注意力掩码
    m_sentence_token: np.ndarray
    m_attention_mask: np.ndarray
    # 输出 一个事件的所有标签
    m_trigger_labels: np.ndarray
    m_subject_object_labels: np.ndarray
    m_time_location_labels: np.ndarray

    # 得到这条文本的 Token,以及 对应的Attention Mask
    # 将每一句转成数字
    # 参考 Transformer 的 encode_plus 文档
    # https://huggingface.co/transformers/internal/tokenization_utils.html#transformers.tokenization_utils_base.PreTrainedTokenizerBase.encode_plus
    # input_ids就是一连串token在字典中的对应id。shape：(Config.batch_size, Config.sequence_length)

    # input_ids = []
    # input_ids.append(tokenizer.encode(test_samples1))
    # input_ids.append(tokenizer.encode(test_samples2))
    # input_ids = torch.LongTensor(input_ids).cuda()

    # attention_masks是对padding部分进行mask，有字部分为1，padding部分为0。shape:(Config.batch_size, Config.sequence_length)
    # attention_masks = torch.LongTensor([[1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
    #                                     [1, 1, 1, 1, 1, 1, 1, 1, 1, 1]]).cuda()
    def encode_sentence(self, tokenizer):
        cur_token = tokenizer.encode_plus(self.m_sentence, padding='max_length',
                                          truncation=True, max_length=self.m_max_length,
                                          return_tensors='np', return_attention_mask=True)
        self.m_sentence_token = cur_token['input_ids'][0]
        self.m_attention_mask = cur_token['attention_mask'][0]
        return

    def encode_arguments(self, tokenizer):
        def encode_one_word(one_word: str, in_tokenizer) -> np.ndarray:
            if one_word:
                cur_word_token = in_tokenizer.encode(one_word,
                                                     add_special_tokens=False,
                                                     return_tensors='np')
                return cur_word_token[0]
            else:
                return np.zeros(shape=(1,))

        self.m_trigger_token = encode_one_word(self.m_trigger, tokenizer)
        self.m_object_token = encode_one_word(self.m_object, tokenizer)
        self.m_subject_token = encode_one_word(self.m_subject, tokenizer)
        self.m_time_token = encode_one_word(self.m_time, tokenizer)
        self.m_location_token = encode_one_word(self.m_location, tokenizer)
        return

    # 定位一个标签的 (起始坐标, 标签长度)
    # 如果 label 为空，返回 (-1,0)
    @staticmethod
    def locate_label_index(sentence_token: np.ndarray, label_token: np.ndarray) -> (int, int):
        def rolling_window(a, size):
            shape = a.shape[:-1] + (a.shape[-1] - size + 1, size)
            strides = a.strides + (a.strides[-1],)
            return np.lib.stride_tricks.as_strided(a, shape=shape, strides=strides)

        if not (np.all(label_token == 0)):
            label_len: int = label_token.shape[0]
            label_is_exist = np.all(rolling_window(sentence_token, label_len) == label_token, axis=1)
            label_index = np.where(label_is_exist == True)
            # 在最大长度中无法定位该标签
            if label_index[0].size == 0:
                return -1,0
            return label_index[0][0], label_len
        else:
            return -1, 0

    # 触发词提取模型测试样例的label。shape：(Config.batch_size, Config.sequence_length, 2)
    # trigger_labels = torch.tensor([[[0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [1, 0], [0, 1], [0, 0], [0, 0], [0, 0]],
    #                                [[0, 0], [0, 0], [0, 0], [1, 0], [0, 1], [0, 0], [0, 0], [0, 0], [0, 0],
    #                                 [0, 0]]]).cuda()
    def set_trigger_label(self):
        trigger_start_idx, trigger_len = Event.locate_label_index(self.m_sentence_token, self.m_trigger_token)
        self.m_trigger_labels = np.zeros((self.m_max_length, 2), dtype=np.int8)

        if trigger_len != 0:
            self.m_trigger_labels[trigger_start_idx][0] = 1
            self.m_trigger_labels[trigger_start_idx + trigger_len - 1][1] = 1
        return

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
    #
    # time_loc_labels = torch.tensor([[0, 1, 2, 5, 7, 0, 0, 0, 0, 0],
    #                                 [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]]).cuda()
    def set_time_loc_label(self):
        time_start_idx, time_len = Event.locate_label_index(self.m_sentence_token, self.m_time_token)
        loc_start_idx, loc_len = Event.locate_label_index(self.m_sentence_token, self.m_location_token)
        self.m_time_location_labels = np.zeros((self.m_max_length), dtype=np.int8)

        # 时间标签
        if time_len != 0:
            if time_len == 1:  # S-time
                self.m_time_location_labels[time_start_idx] = 4
            else:  # BIE-time
                time_I_start_idx = time_start_idx + 1
                time_I_end_idx = time_start_idx + time_len - 2
                # B-time
                self.m_time_location_labels[time_start_idx] = 1
                # I-time
                self.m_time_location_labels[time_I_start_idx: (time_I_end_idx + 1)] = 2
                # E-time
                self.m_time_location_labels[time_start_idx + time_len - 1] = 3
        if loc_len != 0:
            if loc_len == 1:  # S-loc
                self.m_time_location_labels[loc_start_idx] = 8
            else:  # BIE-loc
                loc_I_start_idx = time_start_idx + 1
                loc_I_end_idx = loc_start_idx + loc_len - 2
                # B-loc
                self.m_time_location_labels[loc_start_idx] = 5
                # I-loc
                self.m_time_location_labels[loc_I_start_idx:(loc_I_end_idx + 1)] = 6
                # E-loc
                self.m_time_location_labels[loc_start_idx + loc_len - 1] = 7

        return

    # 主客体识别模型测试样例的label。shape:（Config.train_batch_size, Config.sequence_length, 4）
    # sub_obj_labels = torch.tensor([[[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0],
    #                                 [0, 0, 0, 1], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]],
    #                                [[0, 0, 0, 0], [1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 1, 0],
    #                                 [0, 0, 0, 0], [0, 0, 0, 1], [0, 0, 0, 0], [0, 0, 0, 0]]]).cuda()
    def set_object_subject_label(self):
        object_start_idx, object_len = Event.locate_label_index(self.m_sentence_token, self.m_object_token)
        subject_start_idx, subject_len = Event.locate_label_index(self.m_sentence_token, self.m_subject_token)
        self.m_subject_object_labels = np.zeros((self.m_max_length, 4), dtype=np.int8)

        # [主语起始，主语结束，宾语起始，宾语结束]
        if subject_len != 0:
            self.m_subject_object_labels[subject_start_idx][0] = 1
            self.m_subject_object_labels[subject_start_idx + subject_len - 1][1] = 1
        if object_len != 0:
            self.m_subject_object_labels[object_start_idx][2] = 1
            self.m_subject_object_labels[object_start_idx + object_len - 1][3] = 1

        return

    def __init__(self,
                 sentence: str,
                 trigger: str,
                 object_str: str,
                 subject: str,
                 time: str,
                 loc: str,
                 max_length: int,
                 tokenizer):
        self.m_sentence = sentence
        self.m_trigger = trigger
        self.m_object = object_str
        self.m_subject = subject
        self.m_time = time
        self.m_location = loc

        self.m_max_length = max_length

        self.encode_arguments(tokenizer)
        self.encode_sentence(tokenizer)
        # 设置标签
        self.set_trigger_label()
        self.set_object_subject_label()
        self.set_time_loc_label()
        return


# CorpusData 类，用来读入语料数据、存入语料数据
class CorpusData:

    # 按行读入 json，原数据集有误，每行json之间应该加逗号
    @staticmethod
    def load_json_file(file_path) -> list:
        with open(file_path, 'r') as f:
            result = json.load(f)
        return result

    # 按 key 抽取 json 列表中的元素
    @staticmethod
    def json_list_extract_corpus_text(json_list, sentence_key='text') -> list:
        sentence = []
        for data in json_list:
            sentence.append(data[sentence_key])
        return sentence

    # 从 Json 中读取事件，读取每个论元，生成所有 Event
    # 构造 N 个事件类
    @staticmethod
    def json_list_extract_event(json_list: list,
                                sentence_length,
                                tokenizer,
                                sentence_key='sentence',
                                events_key='events',
                                labels_text_key='text',
                                trigger_key='trigger',
                                arguments_key='arguments',
                                object_key='object',
                                subject_key='subject',
                                time_key='time',
                                location_key='loc', ) -> List[Event]:

        # 从 arguments 列表中抽取元素,
        def extract_event_arguments_from_list(event_arguments_list: list) -> dict:
            event_arguments: dict = {'object': '', 'subject': '', 'time': '', 'loc': ''}
            for one_argument in event_arguments_list:
                cur_argument = one_argument['role']
                if cur_argument in event_arguments:
                    event_arguments[cur_argument] = one_argument['text']
            return event_arguments

        # 返回的结果
        all_events = []
        for one_corpus in json_list:
            # 提取一条语料中的 文本和事件
            one_corpus_sentences = one_corpus[sentence_key]
            one_corpus_events = one_corpus[events_key]
            for one_event in one_corpus_events:
                one_trigger: str = one_event[trigger_key][labels_text_key]
                one_event_arguments: list = one_event[arguments_key]
                cur_arguments: dict = extract_event_arguments_from_list(one_event_arguments)

                one_event = Event(one_corpus_sentences, one_trigger,
                                  cur_arguments[object_key], cur_arguments[subject_key],
                                  cur_arguments[time_key], cur_arguments[location_key],
                                  sentence_length, tokenizer)
                all_events.append(one_event)

        return all_events

    # 得到所有事件中的 sentence_token
    @staticmethod
    def extract_all_sentences_token_from_events(all_events: List[Event]) -> np.ndarray:
        all_events_len: int = len(all_events)
        if all_events_len == 0:
            return np.zeros(1)
        else:
            # 获取一句话的长度，和数据类型
            sequence_length: int = all_events[0].m_max_length
            token_data_type: np.dtype = all_events[0].m_sentence_token.dtype
            all_sentences_token = np.zeros((all_events_len, sequence_length), dtype=token_data_type)

            for idx in range(all_events_len):
                all_sentences_token[idx] = all_events[idx].m_sentence_token
            return all_sentences_token
    # 得到所有事件中的 attention mask
    @staticmethod
    def extract_all_attention_mask_from_events(all_events: List[Event]) -> np.ndarray:
        all_events_len: int = len(all_events)
        if all_events_len == 0:
            return np.zeros(1)
        else:
            # 获取一句话的长度，和数据类型
            sequence_length: int = all_events[0].m_max_length
            token_data_type: np.dtype = all_events[0].m_attention_mask.dtype
            all_attention_mask = np.zeros((all_events_len, sequence_length), dtype=token_data_type)

            for idx in range(all_events_len):
                all_attention_mask[idx] = all_events[idx].m_attention_mask
            return all_attention_mask

    # 得到所有事件中的 trigger_label
    @staticmethod
    def extract_all_trigger_labels_from_events(all_events: List[Event]) -> np.ndarray:
        all_events_len: int = len(all_events)
        if all_events_len == 0:
            return np.zeros(1)
        else:
            # 获取一句话的长度，和数据类型
            sequence_length: int = all_events[0].m_max_length
            token_data_type: np.dtype = all_events[0].m_trigger_labels.dtype
            # 一个字对应 2 维向量
            all_trigger_labels = np.zeros((all_events_len, sequence_length, 2), dtype=token_data_type)

            for idx in range(all_events_len):
                all_trigger_labels[idx] = all_events[idx].m_trigger_labels
            return all_trigger_labels

    # 得到所有事件中的 主语宾语
    @staticmethod
    def extract_all_subject_object_labels_from_events(all_events: List[Event]) -> np.ndarray:
        all_events_len: int = len(all_events)
        if all_events_len == 0:
            return np.zeros(1)
        else:
            # 获取一句话的长度，和数据类型
            sequence_length: int = all_events[0].m_max_length
            token_data_type: np.dtype = all_events[0].m_subject_object_labels.dtype
            # 一个字对应4 维向量
            all_sub_obj_labels = np.zeros((all_events_len, sequence_length, 4), dtype=token_data_type)

            for idx in range(all_events_len):
                all_sub_obj_labels[idx] = all_events[idx].m_subject_object_labels
            return all_sub_obj_labels

    # 得到所有事件中的 时间地点
    @staticmethod
    def extract_all_time_location_labels_from_events(all_events: List[Event]) -> np.ndarray:
        all_events_len: int = len(all_events)
        if all_events_len == 0:
            return np.zeros(1)
        else:
            # 获取一句话的长度，和数据类型
            sequence_length: int = all_events[0].m_max_length
            token_data_type: np.dtype = all_events[0].m_time_location_labels.dtype
            # 一个字一个值
            all_time_loc_labels = np.zeros((all_events_len, sequence_length), dtype=token_data_type)

            for idx in range(all_events_len):
                all_time_loc_labels[idx] = all_events[idx].m_time_location_labels
            return all_time_loc_labels

    # 外部调用这个函数！！！
    # 读取数据的完整函数！！！！返回一个 dict, {'input_ids': 输入词对应token id，'attention_masks': }
    # corpus_data = {'input_ids': input_ids,
    #                 'attention_masks': attention_masks,
    #                 'trigger_labels': trigger_labels,
    #                 'sub_obj_labels': sub_obj_labels,
    #                 'time_loc_labels': time_loc_labels}
    @staticmethod
    def load_corpus_data(file_name: str = '', load_data_from_cache: bool = False,
                         cached_file_name: str = '') -> dict:

        # 从数据文件中直接读取多个数组
        if load_data_from_cache:
            cached_file_name = cached_file_name + '.npy'
            corpus_data = np.load(cached_file_name, allow_pickle=True)
            print("数据加载完毕")
            return corpus_data
        # 从Json FILE中读取数组
        else:
            data_set_json = CorpusData.load_json_file(file_name)
            # 设置 tokenizer
            tokenizer = BertTokenizer.from_pretrained(
                "hfl/chinese-roberta-wwm-ext", cache_dir="../saved_model/transformer_cached")
            # 读入所有事件
            all_events:list = CorpusData.json_list_extract_event(data_set_json, Config.sequence_length, tokenizer)

            input_ids = CorpusData.extract_all_sentences_token_from_events(all_events)
            attention_masks = CorpusData.extract_all_attention_mask_from_events(all_events)
            trigger_labels = CorpusData.extract_all_trigger_labels_from_events(all_events)
            sub_obj_labels = CorpusData.extract_all_subject_object_labels_from_events(all_events)
            time_loc_labels = CorpusData.extract_all_time_location_labels_from_events(all_events)

            # 构造一个 dict
            corpus_data = {'input_ids': input_ids,
                           'attention_masks': attention_masks,
                           'trigger_labels': trigger_labels,
                           'sub_obj_labels': sub_obj_labels,
                           'time_loc_labels': time_loc_labels}
            print("数据加载完毕")
            return corpus_data




    # https://numpy.org/doc/stable/reference/generated/numpy.load.html
    # 必须指明 allow_pickle = True
    # 保存 dict of numpy.ndarray
    @staticmethod
    def save_corpus_data(all_corpus_data: dict, output_file_name: str):
        np.save(output_file_name, all_corpus_data, allow_pickle=True)
        print("数据储存完毕")
        return

    '''
        以下函数暂时忽略
    '''

    @staticmethod
    def json_list_extract_corpus_label(json_list, label_key) -> list:
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
    def convert_text_to_token_ndarray(tokenizer, all_sentences: list, limit_size=Config.sequence_length) -> (
    np.ndarray, np.ndarray):
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
    def convert_labels_to_mask_ndarray(tokenizer, all_sentences_labels: list,
                                       all_sentences_tokens: np.ndarray) -> np.ndarray:
        def rolling_window(a, size):
            shape = a.shape[:-1] + (a.shape[-1] - size + 1, size)
            strides = a.strides + (a.strides[-1],)
            return np.lib.stride_tricks.as_strided(a, shape=shape, strides=strides)

        all_lables_tokens: list = CorpusData.convert_labels_to_tokens(
            tokenizer, all_sentences_labels)

        # 创建 标签掩码矩阵 shape:(样本总个数, Config.sequence_length, 2)
        all_lables_mask: np.ndarray = np.zeros(
            (all_sentences_tokens.shape[0], all_sentences_tokens.shape[1], 2), dtype=np.int8)
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

                # 每个字有两个值，第一个值代表这个字是不是trigger的start，第二个值代表这个字是不是trigger的end
                all_lables_mask[idx][label_index][0] = 1
                all_lables_mask[idx][label_index + cur_label_len - 1][1] = 1

        return all_lables_mask
        # return triggers_tensor

    # 外部调用这个函数！！！
    # 读取数据的完整函数！！！！返回一个 dict, {'input_ids': 输入词对应token id，'attention_masks': }

    @staticmethod
    def load_corpus_data_old(file_name: str = '', load_data_from_cache: bool = False,
                             cached_file_name: str = '') -> dict:

        # 从数据文件中直接读取多个数组
        if load_data_from_cache:
            cached_file_name = cached_file_name + '.npy'
            corpus_data = np.load(cached_file_name, allow_pickle=True)
            print("数据加载完毕")
            return corpus_data
        # 从Json FILE中读取数组
        else:
            data_set_json = CorpusData.load_json_file(file_name)
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
                           'trigger_labels': all_sentences_triggers_labels,
                           'object_labels': all_sentences_objects_labels,
                           'subject_labels': all_sentences_subjects_labels}
            return corpus_data




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
    raw_data_path = '../dataset/xf_2020_Corpus/final/raw_data/preliminary_data_pred_trigger_and_role.json'
    sentence_set = CorpusData.load_json_file(raw_data_path)
    sentences = CorpusData.json_list_extract_corpus_text(sentence_set, sentence_key='sentence')

    print(sentences[0])

    # tokenizer = BertTokenizer.from_pretrained(
    #     "hfl/chinese-roberta-wwm-ext", cache_dir="../saved_model/transformer_cached")
    # max_length = Config.sequence_length

    # all_events = CorpusData.json_list_extract_event(sentence_set, max_length, tokenizer)
    corpus_data = CorpusData.load_corpus_data(raw_data_path)
    CorpusData.save_corpus_data(corpus_data,'../saved_model/preliminary_data_pred_trigger_and_role')
    corpus_data_from_saved = CorpusData.load_corpus_data(load_data_from_cache=True,
                                                         cached_file_name='../saved_model/preliminary_data_pred_trigger_and_role')
    print(0)
    # 这里的 main 函数仅作测试用，具体参数变量参考 Config 文件
    # dev_data_path = '../dataset/xf_2020_data/preliminary/raw_data/dev.json'

# {'input_ids': array([[ 101,  100, 3209, ...,    0,    0,    0],
#         [ 101,  125,  119, ...,    0,    0,    0],
#         [ 101,  125,  119, ...,    0,    0,    0],
#         ...,
#         [ 101, 3189, 3315, ...,    0,    0,    0],
#         [ 101, 3209, 3633, ...,    0,    0,    0],
#         [ 101, 5401, 1744, ...,    0,    0,    0]]),
#  'attention_masks': array([[1, 1, 1, ..., 0, 0, 0],
#         [1, 1, 1, ..., 0, 0, 0],
#         [1, 1, 1, ..., 0, 0, 0],
#         ...,
#         [1, 1, 1, ..., 0, 0, 0],
#         [1, 1, 1, ..., 0, 0, 0],
#         [1, 1, 1, ..., 0, 0, 0]]),
#  'trigger_labels': array([[[0, 0],
#          [0, 0],
#          [0, 0],
#          ...,
#
#         [[0, 0],
#          [0, 0],
#          [0, 0],
#          ...,
#          [0, 0],
#          [0, 0],
#          [0, 0]]], dtype=int8),
#  'sub_obj_labels': array([[[0, 0, 0, 0],
#          [0, 0, 0, 0],
#          [0, 0, 0, 0],
#          ...,
#          ...,
#          [0, 0, 0, 0],
#          [0, 0, 0, 0],
#          [0, 0, 0, 0]]], dtype=int8),
#  'time_loc_labels': array([[6, 6, 6, ..., 0, 0, 0],
#         [0, 0, 0, ..., 0, 0, 0],
#         [0, 0, 0, ..., 0, 0, 0],
#         ...,
#         [0, 0, 0, ..., 0, 0, 0],
#         [0, 0, 0, ..., 0, 0, 0],
#         [0, 0, 0, ..., 0, 0, 0]], dtype=int8)}