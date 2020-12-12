import numpy as np

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
    m_trigger_index: np.ndarray
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
                return -1, 0
            return label_index[0][0], label_len
        else:
            return -1, 0

    # 触发词提取模型测试样例的label。shape：(Config.batch_size, Config.sequence_length, 2)
    # trigger_labels = torch.tensor([[[0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [1, 0], [0, 1], [0, 0], [0, 0], [0, 0]],
    #                                [[0, 0], [0, 0], [0, 0], [1, 0], [0, 1], [0, 0], [0, 0], [0, 0], [0, 0],
    #                                 [0, 0]]]).cuda()
    def set_trigger_label_and_index(self):
        trigger_start_idx, trigger_len = Event.locate_label_index(self.m_sentence_token, self.m_trigger_token)
        self.m_trigger_labels = np.zeros((self.m_max_length, 2), dtype=np.int32)
        self.m_trigger_index = np.zeros(shape=(2,), dtype=np.int32)

        if trigger_len != 0:
            self.m_trigger_labels[trigger_start_idx][0] = 1
            self.m_trigger_labels[trigger_start_idx + trigger_len - 1][1] = 1

            self.m_trigger_index[0] = trigger_start_idx
            self.m_trigger_index[1] = trigger_start_idx + trigger_len - 1
            # print(self.m_trigger)
            # print(self.m_trigger_index)
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
        self.m_time_location_labels = np.zeros((self.m_max_length), dtype=np.int32)

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
        self.m_subject_object_labels = np.zeros((self.m_max_length, 4), dtype=np.int32)

        # [主语起始，主语结束，宾语起始，宾语结束]
        if subject_len != 0:
            self.m_subject_object_labels[subject_start_idx][0] = 1
            self.m_subject_object_labels[subject_start_idx + subject_len - 1][1] = 1
        if object_len != 0:
            self.m_subject_object_labels[object_start_idx][2] = 1
            self.m_subject_object_labels[object_start_idx + object_len - 1][3] = 1

        return

    # 如果是训练 + 测试集，则 encode 句子 + 注意力掩码 + 论元位置标签
    # 如果是预测集，仅encode 句子 + 注意力掩码
    def __init__(self,
                 tokenizer,
                 sentence='',
                 trigger='',
                 object_str='',
                 subject='',
                 time='',
                 loc='',
                 max_length=0,

                 is_predict=False):
        self.m_sentence = sentence
        self.m_trigger = trigger
        self.m_object = object_str
        self.m_subject = subject
        self.m_time = time
        self.m_location = loc

        self.m_max_length = max_length

        self.encode_sentence(tokenizer)
        if not is_predict:
            self.encode_arguments(tokenizer)
            # 设置标签
            self.set_trigger_label_and_index()
            self.set_object_subject_label()
            self.set_time_loc_label()

        return
